import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import math

from room_environment import create_room


class NavigationEnv(gym.Env):
    """
    A robot learns to navigate to a goal without hitting walls.
    
    Observation: [robot_x, robot_y, robot_angle, dist_to_goal, angle_to_goal, dist_to_wall]
    Actions: 0=forward, 1=turn_left, 2=turn_right
    """
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Render mode: "human" shows GUI, None runs headless (faster for training)
        self.render_mode = render_mode
        
        # Connect to PyBullet
        if render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)  # Headless mode
        
        # Set up environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load floor and room
        self.plane_id = p.loadURDF("plane.urdf")
        self.room_size = 10
        self.walls = create_room(room_size=self.room_size, wall_height=2, wall_thickness=0.1)
        
        # Load robot
        self.robot_id = p.loadURDF(
            "husky/husky.urdf",
            [0, 0, 0.2],
            p.getQuaternionFromEuler([0, 0, 0])
        )
        
        # Wheel joints (Husky robot)
        self.front_left = 2
        self.front_right = 3
        self.rear_left = 4
        self.rear_right = 5
        
        # Movement parameters
        self.forward_velocity = 10
        self.turn_velocity = 5
        
        # Environment parameters
        self.max_steps = 500
        self.goal_threshold = 0.8  # meters - close enough to goal
        self.step_count = 0
        
        # Goal position (will be set in reset)
        self.goal_position = [0, 0]
        self.previous_distance = 0
        
        # Define action space: 0=forward, 1=turn_left, 2=turn_right
        self.action_space = spaces.Discrete(3)
        
        # Define observation space
        # [robot_x, robot_y, robot_angle, dist_to_goal, angle_to_goal, dist_to_wall]
        self.observation_space = spaces.Box(
            low=np.array([-20, -20, -np.pi, 0, -np.pi, 0], dtype=np.float32),
            high=np.array([20, 20, np.pi, 30, np.pi, 20], dtype=np.float32),
            dtype=np.float32
        )
        
        # Set camera if in GUI mode
        if render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=12,
                cameraYaw=0,
                cameraPitch=-60,
                cameraTargetPosition=[0, 0, 0]
            )
    
    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        super().reset(seed=seed)
        
        # Reset step counter
        self.step_count = 0
        
        # Reset robot position to center with random orientation
        random_angle = np.random.uniform(-np.pi, np.pi)
        p.resetBasePositionAndOrientation(
            self.robot_id,
            [0, 0, 0.2],
            p.getQuaternionFromEuler([0, 0, random_angle])
        )
        
        # Stop all wheels
        self._set_wheel_velocities(0, 0)
        
        # Set random goal position (inside room, away from walls and center)
        margin = 1.5  # Stay away from walls
        while True:
            goal_x = np.random.uniform(-self.room_size/2 + margin, self.room_size/2 - margin)
            goal_y = np.random.uniform(-self.room_size/2 + margin, self.room_size/2 - margin)
            
            # Make sure goal is not too close to start
            if math.sqrt(goal_x**2 + goal_y**2) > 2.0:
                break
        
        self.goal_position = [goal_x, goal_y]
        
        # Draw goal marker if in GUI mode
        if self.render_mode == "human":
            self._draw_goal_marker()
        
        # Calculate initial distance to goal
        observation = self._get_observation()
        self.previous_distance = self._get_distance_to_goal()
        
        return observation, {}
    
    def step(self, action):
        """Execute one action and return results."""
        self.step_count += 1
        
        # Execute action
        if action == 0:  # Forward
            self._set_wheel_velocities(self.forward_velocity, self.forward_velocity)
        elif action == 1:  # Turn left
            self._set_wheel_velocities(-self.turn_velocity, self.turn_velocity)
        elif action == 2:  # Turn right
            self._set_wheel_velocities(self.turn_velocity, -self.turn_velocity)
        
        # Run simulation for a short time
        for _ in range(10):
            p.stepSimulation()
        
        # Stop wheels
        self._set_wheel_velocities(0, 0)
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = -0.1  # Small penalty for each step (encourages efficiency)
        
        distance_to_goal = self._get_distance_to_goal()
        
        # Check if reached goal
        if distance_to_goal < self.goal_threshold:
            reward = 100.0
            done = True
            truncated = False
        # Check if hit wall
        elif self._check_wall_collision():
            reward = -100.0
            done = True
            truncated = False
        # Check if timed out
        elif self.step_count >= self.max_steps:
            done = False
            truncated = True
        else:
            # Reward for getting closer to goal
            reward += (self.previous_distance - distance_to_goal) * 10
            done = False
            truncated = False
        
        self.previous_distance = distance_to_goal
        
        return observation, reward, done, truncated, {}
    
    def _set_wheel_velocities(self, left_vel, right_vel):
        """Set velocities for left and right wheels."""
        p.setJointMotorControl2(self.robot_id, self.front_left, p.VELOCITY_CONTROL, targetVelocity=left_vel)
        p.setJointMotorControl2(self.robot_id, self.rear_left, p.VELOCITY_CONTROL, targetVelocity=left_vel)
        p.setJointMotorControl2(self.robot_id, self.front_right, p.VELOCITY_CONTROL, targetVelocity=right_vel)
        p.setJointMotorControl2(self.robot_id, self.rear_right, p.VELOCITY_CONTROL, targetVelocity=right_vel)
    
    def _get_observation(self):
        """Get current observation."""
        # Get robot position and orientation
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        robot_x, robot_y = position[0], position[1]
        
        # Convert orientation to angle
        euler = p.getEulerFromQuaternion(orientation)
        robot_angle = euler[2]  # Yaw angle
        
        # Distance to goal
        dist_to_goal = self._get_distance_to_goal()
        
        # Angle to goal
        dx = self.goal_position[0] - robot_x
        dy = self.goal_position[1] - robot_y
        angle_to_goal = math.atan2(dy, dx) - robot_angle
        
        # Normalize angle to [-pi, pi]
        while angle_to_goal > np.pi:
            angle_to_goal -= 2 * np.pi
        while angle_to_goal < -np.pi:
            angle_to_goal += 2 * np.pi
        
        # Distance to nearest wall (simple approximation)
        dist_to_wall = self._get_distance_to_nearest_wall(robot_x, robot_y)
        
        return np.array([robot_x, robot_y, robot_angle, dist_to_goal, angle_to_goal, dist_to_wall], dtype=np.float32)
    
    def _get_distance_to_goal(self):
        """Calculate distance from robot to goal."""
        position, _ = p.getBasePositionAndOrientation(self.robot_id)
        dx = self.goal_position[0] - position[0]
        dy = self.goal_position[1] - position[1]
        return math.sqrt(dx**2 + dy**2)
    
    def _get_distance_to_nearest_wall(self, x, y):
        """Get distance to nearest wall (simple calculation)."""
        half_size = self.room_size / 2
        distances = [
            half_size - x,   # East wall
            half_size + x,   # West wall
            half_size - y,   # North wall
            half_size + y    # South wall
        ]
        return min(distances)
    
    def _check_wall_collision(self):
        """Check if robot has collided with any wall."""
        for wall_id in self.walls:
            contacts = p.getContactPoints(self.robot_id, wall_id)
            if len(contacts) > 0:
                return True
        return False
    
    def _draw_goal_marker(self):
        """Draw a visual marker at the goal position."""
        # Remove old markers
        p.removeAllUserDebugItems()
        
        # Draw goal as a sphere
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.3,
            rgbaColor=[0, 1, 0, 0.7]  # Green
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=[self.goal_position[0], self.goal_position[1], 0.3]
        )
    
    def close(self):
        """Clean up."""
        p.disconnect()


# Test the environment
if __name__ == "__main__":
    print("Testing NavigationEnv...")
    
    # Create environment with GUI
    env = NavigationEnv(render_mode="human")
    
    # Reset
    observation, info = env.reset()
    print(f"Initial observation: {observation}")
    print(f"Goal position: {env.goal_position}")
    
    # Run a few random actions
    print("\nRunning 100 random actions...")
    for i in range(100):
        action = env.action_space.sample()  # Random action
        observation, reward, done, truncated, info = env.step(action)
        
        if done or truncated:
            print(f"Episode ended at step {i+1}, reward: {reward}")
            observation, info = env.reset()
    
    print("\nTest complete. Close the window to exit.")
    
    # Keep window open
    import time
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1/240)
    
    env.close()