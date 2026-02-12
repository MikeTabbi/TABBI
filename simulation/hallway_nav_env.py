import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import math
import random

from hallway_environment import create_hallway_environment, get_random_hallway_position


class HallwayNavEnv(gym.Env):
    """
    Robot learns to navigate hallways with intersections.
    
    Observation: [robot_x, robot_y, robot_angle, dist_to_goal, angle_to_goal, 
                  dist_front, dist_left, dist_right]
    Actions: 0=forward, 1=turn_left, 2=turn_right
    """
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        
        # how pybullet connects
        if render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # env setup
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # floor
        self.plane_id = p.loadURDF("plane.urdf")
        
        # making a hallway
        self.env_data = create_hallway_environment(env_size=20, hallway_width=3)
        self.walls = self.env_data['walls']
        self.valid_positions = self.env_data['valid_positions']
        
        # spot jr
        self.robot_id = p.loadURDF(
            "husky/husky.urdf",
            [0, 0, 0.2],
            p.getQuaternionFromEuler([0, 0, 0])
        )
        
        # spot jr's wheel joints
        self.front_left = 2
        self.front_right = 3
        self.rear_left = 4
        self.rear_right = 5
        
        # spot jr's movement parameters
        self.forward_velocity = 10
        self.turn_velocity = 5
        
        # parameters of the environment
        self.max_steps = 1000
        self.goal_threshold = 1.0
        self.step_count = 0
        
        # spot jr's goal
        self.goal_position = [0, 0]
        self.previous_distance = 0
        
        # action space: 0=forward, 1=turn_left, 2=turn_right
        self.action_space = spaces.Discrete(3)
        
        # observation space
        # [robot_x, robot_y, robot_angle, dist_to_goal, angle_to_goal, dist_front, dist_left, dist_right]
        self.observation_space = spaces.Box(
            low=np.array([-15, -15, -np.pi, 0, -np.pi, 0, 0, 0], dtype=np.float32),
            high=np.array([15, 15, np.pi, 30, np.pi, 10, 10, 10], dtype=np.float32),
            dtype=np.float32
        )
        
        # camera
        if render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=25,
                cameraYaw=0,
                cameraPitch=-89,
                cameraTargetPosition=[0, 0, 0]
            )
    
    def reset(self, seed=None, options=None):
        """Reset environment for new episode."""
        super().reset(seed=seed)
        
        self.step_count = 0
        
        # random start position
        start_pos = get_random_hallway_position(self.valid_positions)
        start_angle = random.uniform(-np.pi, np.pi)
        
        p.resetBasePositionAndOrientation(
            self.robot_id,
            start_pos,
            p.getQuaternionFromEuler([0, 0, start_angle])
        )
        
        # wheels stop
        self._set_wheel_velocities(0, 0)
        
        # random goal
        while True:
            goal_pos = get_random_hallway_position(self.valid_positions)
            dist = math.sqrt((goal_pos[0] - start_pos[0])**2 + (goal_pos[1] - start_pos[1])**2)
            if dist > 5.0:  # Goal must be at least 5m away
                break
        
        self.goal_position = [goal_pos[0], goal_pos[1]]
        
        # Draw goal marker
        if self.render_mode == "human":
            self._draw_goal_marker()
        
        # Get initial observation
        observation = self._get_observation()
        self.previous_distance = self._get_distance_to_goal()
        
        return observation, {}
    
    def step(self, action):
        """Execute action and return results."""
        self.step_count += 1
        
        # Execute action
        if action == 0:  # Forward
            self._set_wheel_velocities(self.forward_velocity, self.forward_velocity)
        elif action == 1:  # Turn left
            self._set_wheel_velocities(-self.turn_velocity, self.turn_velocity)
        elif action == 2:  # Turn right
            self._set_wheel_velocities(self.turn_velocity, -self.turn_velocity)
        
        # Step simulation
        for _ in range(10):
            p.stepSimulation()
        
        # Stop wheels
        self._set_wheel_velocities(0, 0)
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = -0.1  # Small step penalty
        
        distance_to_goal = self._get_distance_to_goal()
        
        # Check if reached goal
        if distance_to_goal < self.goal_threshold:
            reward = 100.0
            done = True
            truncated = False
        # Check wall collision
        elif self._check_wall_collision():
            reward = -50.0
            done = True
            truncated = False
        # Check timeout
        elif self.step_count >= self.max_steps:
            done = False
            truncated = True
        else:
            # Reward for getting closer
            reward += (self.previous_distance - distance_to_goal) * 5
            done = False
            truncated = False
        
        self.previous_distance = distance_to_goal
        
        return observation, reward, done, truncated, {}
    
    def _set_wheel_velocities(self, left_vel, right_vel):
        """Set wheel velocities."""
        p.setJointMotorControl2(self.robot_id, self.front_left, p.VELOCITY_CONTROL, targetVelocity=left_vel)
        p.setJointMotorControl2(self.robot_id, self.rear_left, p.VELOCITY_CONTROL, targetVelocity=left_vel)
        p.setJointMotorControl2(self.robot_id, self.front_right, p.VELOCITY_CONTROL, targetVelocity=right_vel)
        p.setJointMotorControl2(self.robot_id, self.rear_right, p.VELOCITY_CONTROL, targetVelocity=right_vel)
    
    def _get_observation(self):
        """Get current observation."""
        # Robot position and orientation
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        robot_x, robot_y = position[0], position[1]
        euler = p.getEulerFromQuaternion(orientation)
        robot_angle = euler[2]
        
        # Distance and angle to goal
        dist_to_goal = self._get_distance_to_goal()
        dx = self.goal_position[0] - robot_x
        dy = self.goal_position[1] - robot_y
        angle_to_goal = math.atan2(dy, dx) - robot_angle
        
        # Normalize angle
        while angle_to_goal > np.pi:
            angle_to_goal -= 2 * np.pi
        while angle_to_goal < -np.pi:
            angle_to_goal += 2 * np.pi
        
        # Distance sensors (ray casting)
        dist_front = self._cast_ray(robot_x, robot_y, robot_angle)
        dist_left = self._cast_ray(robot_x, robot_y, robot_angle + np.pi/2)
        dist_right = self._cast_ray(robot_x, robot_y, robot_angle - np.pi/2)
        
        return np.array([
            robot_x, robot_y, robot_angle,
            dist_to_goal, angle_to_goal,
            dist_front, dist_left, dist_right
        ], dtype=np.float32)
    
    def _cast_ray(self, x, y, angle, max_dist=10.0):
        """Cast a ray and return distance to nearest obstacle."""
        end_x = x + max_dist * math.cos(angle)
        end_y = y + max_dist * math.sin(angle)
        
        result = p.rayTest([x, y, 0.5], [end_x, end_y, 0.5])[0]
        hit_fraction = result[2]
        
        return hit_fraction * max_dist
    
    def _get_distance_to_goal(self):
        """Get distance from robot to goal."""
        position, _ = p.getBasePositionAndOrientation(self.robot_id)
        dx = self.goal_position[0] - position[0]
        dy = self.goal_position[1] - position[1]
        return math.sqrt(dx**2 + dy**2)
    
    def _check_wall_collision(self):
        """Check if robot hit a wall."""
        for wall_id in self.walls:
            contacts = p.getContactPoints(self.robot_id, wall_id)
            if contacts is not None and len(contacts) > 0:
                return True
        return False
    
    def _draw_goal_marker(self):
        """Draw goal marker."""
        p.removeAllUserDebugItems()
        visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.5,
            rgbaColor=[0, 1, 0, 0.7]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual,
            basePosition=[self.goal_position[0], self.goal_position[1], 0.5]
        )
    
    def close(self):
        """Clean up."""
        if p.isConnected():
            p.disconnect()


# Test the environment
if __name__ == "__main__":
    print("Testing HallwayNavEnv...")
    
    env = HallwayNavEnv(render_mode="human")
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation: {obs}")
    print(f"Goal: {env.goal_position}")
    
    print("\nRunning 200 random steps...")
    for i in range(200):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        if done or truncated:
            print(f"Episode ended at step {i+1}, reward: {reward:.1f}")
            obs, info = env.reset()
    
    print("\nTest complete. Close window to exit.")
    
    import time
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1/240)
    
    env.close()