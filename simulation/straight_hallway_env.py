import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import math
import random


def create_wall(position, half_extents, color=[0.8, 0.8, 0.8, 1]):
    """Create a wall."""
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents
    )
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color
    )
    wall_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position
    )
    return wall_id


class StraightHallwayEnv(gym.Env):
    """
    Simple straight hallway - the easiest navigation task.
    Robot starts at one end, goal is at the other end.
    
    IMPROVED VERSION with better reward tuning for hallway navigation.
    """
    
    def __init__(self, render_mode=None, hallway_length=12, hallway_width=3):
        super().__init__()
        
        self.render_mode = render_mode
        self.hallway_length = hallway_length
        self.hallway_width = hallway_width
        
        # Connect to PyBullet
        if render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Set up environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load floor
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Create hallway
        self.walls = self._create_hallway()
        
        # Load robot
        self.robot_id = p.loadURDF(
            "husky/husky.urdf",
            [0, 0, 0.2],
            p.getQuaternionFromEuler([0, 0, 0])
        )
        
        # Wheel joints
        self.front_left = 2
        self.front_right = 3
        self.rear_left = 4
        self.rear_right = 5
        
        # Movement parameters
        self.forward_velocity = 10
        self.turn_velocity = 5
        
        # Environment parameters
        self.max_steps = 1000  # INCREASED from 500 - give more time to learn
        self.goal_threshold = 1.0
        self.step_count = 0
        
        # Goal
        self.goal_position = [0, 0]
        self.previous_distance = 0
        
        # Milestone tracking (for bonus rewards)
        self.halfway_bonus_given = False
        
        # Action space: 0=forward, 1=turn_left, 2=turn_right
        self.action_space = spaces.Discrete(3)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=np.array([-15, -15, -np.pi, 0, -np.pi, 0, 0, 0], dtype=np.float32),
            high=np.array([15, 15, np.pi, 30, np.pi, 10, 10, 10], dtype=np.float32),
            dtype=np.float32
        )
        
        # Camera setup
        if render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=15,
                cameraYaw=90,
                cameraPitch=-60,
                cameraTargetPosition=[0, 0, 0]
            )
    
    def _create_hallway(self):
        """Create a simple straight hallway."""
        walls = []
        wall_height = 2.0
        half_height = wall_height / 2
        half_length = self.hallway_length / 2
        half_width = self.hallway_width / 2
        thickness = 0.1
        
        # Left wall
        walls.append(create_wall(
            position=[0, half_width, half_height],
            half_extents=[half_length, thickness, half_height]
        ))
        
        # Right wall
        walls.append(create_wall(
            position=[0, -half_width, half_height],
            half_extents=[half_length, thickness, half_height]
        ))
        
        # Back wall (start end)
        walls.append(create_wall(
            position=[-half_length, 0, half_height],
            half_extents=[thickness, half_width, half_height]
        ))
        
        # Front wall (goal end)
        walls.append(create_wall(
            position=[half_length, 0, half_height],
            half_extents=[thickness, half_width, half_height]
        ))
        
        return walls
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.halfway_bonus_given = False  # Reset milestone tracker
        
        # Robot starts at back of hallway, facing forward
        start_x = -self.hallway_length / 2 + 1.5
        start_y = random.uniform(-0.5, 0.5)  # Small random offset
        start_angle = random.uniform(-0.2, 0.2)  # Small random angle
        
        p.resetBasePositionAndOrientation(
            self.robot_id,
            [start_x, start_y, 0.2],
            p.getQuaternionFromEuler([0, 0, start_angle])
        )
        
        # Stop wheels
        self._set_wheel_velocities(0, 0)
        
        # Goal is at front of hallway
        goal_x = self.hallway_length / 2 - 1.5
        goal_y = 0
        self.goal_position = [goal_x, goal_y]
        
        # Draw goal marker
        if self.render_mode == "human":
            self._draw_goal_marker()
        
        # Get initial observation
        observation = self._get_observation()
        self.previous_distance = self._get_distance_to_goal()
        
        return observation, {}
    
    def step(self, action):
        """Execute action."""
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
        distance_to_goal = self._get_distance_to_goal()
        
        # IMPROVED REWARD FUNCTION
        reward = 0
        
        # Big reward for reaching goal
        if distance_to_goal < self.goal_threshold:
            reward = 500.0  # INCREASED from 200
            done = True
            truncated = False
        # Strong penalty for hitting wall
        elif self._check_wall_collision():
            reward = -100.0  # INCREASED from -50
            done = True
            truncated = False
        # Timeout penalty
        elif self.step_count >= self.max_steps:
            reward = -50.0  # INCREASED from -20
            done = False
            truncated = True
        else:
            # STRONG reward for progress toward goal
            progress = self.previous_distance - distance_to_goal
            reward = progress * 100  # INCREASED from 20
            
            # Bonus for reaching halfway point (encourages long-term progress)
            if distance_to_goal < self.hallway_length / 2 and not self.halfway_bonus_given:
                reward += 100.0
                self.halfway_bonus_given = True
            
            # REMOVED action-specific bonuses/penalties
            # Let the robot explore turning strategies freely
            
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
        
        # Distance sensors
        dist_front = self._cast_ray(robot_x, robot_y, robot_angle)
        dist_left = self._cast_ray(robot_x, robot_y, robot_angle + np.pi/2)
        dist_right = self._cast_ray(robot_x, robot_y, robot_angle - np.pi/2)
        
        return np.array([
            robot_x, robot_y, robot_angle,
            dist_to_goal, angle_to_goal,
            dist_front, dist_left, dist_right
        ], dtype=np.float32)
    
    def _cast_ray(self, x, y, angle, max_dist=10.0):
        """Cast a ray and return distance to obstacle."""
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
    print("Testing StraightHallwayEnv (IMPROVED VERSION)...")
    print("\nKey improvements:")
    print("- Goal reward: 200 -> 500")
    print("- Wall penalty: -50 -> -100")
    print("- Progress reward: 20x -> 100x")
    print("- Halfway bonus: +100")
    print("- Max steps: 500 -> 1000")
    print("- Removed turn penalties")
    print()
    
    env = StraightHallwayEnv(render_mode="human")
    
    obs, info = env.reset()
    print(f"Observation: {obs}")
    print(f"Goal: {env.goal_position}")
    print(f"Distance to goal: {env._get_distance_to_goal():.2f}m")
    
    print("\nRunning 100 random steps...")
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        if done or truncated:
            print(f"Episode ended at step {i+1}")
            print(f"Total reward: {total_reward:.1f}")
            print(f"Final distance: {env._get_distance_to_goal():.2f}m")
            obs, info = env.reset()
            total_reward = 0
    
    print("\nTest complete. Close window to exit.")
    
    import time
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1/240)
    
    env.close()