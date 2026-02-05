import sys
import os
import time
import numpy as np

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from stable_baselines3 import PPO
from navigation_env import NavigationEnv
from src.exploration.frontier_explorer import FrontierExplorer
import pybullet as p


class AutonomousExplorer:
    """
    Combines frontier exploration with trained navigation policy
    for fully autonomous exploration.
    """
    
    def __init__(self, model_path="simulation/navigation_policy"):
        print("=" * 50)
        print("Autonomous Explorer")
        print("=" * 50)
        
        # Create environment with GUI
        print("\nCreating environment...")
        self.env = NavigationEnv(render_mode="human")
        
        # Load trained navigation policy
        print("Loading trained navigation policy...")
        self.policy = PPO.load(model_path)
        
        # Create frontier explorer
        print("Creating frontier explorer...")
        self.explorer = FrontierExplorer(map_size=10.0, resolution=0.5)
        
        # Settings
        self.sensor_range = 2.0
        self.max_steps = 2000
        self.steps_per_frontier = 50  # Steps to take toward each frontier
        
        print("\nReady for autonomous exploration!")
    
    def get_robot_position(self):
        """Get current robot position from environment."""
        position, _ = p.getBasePositionAndOrientation(self.env.robot_id)
        return position[0], position[1]
    
    def run(self):
        """Run autonomous exploration."""
        print("\n" + "=" * 50)
        print("Starting autonomous exploration...")
        print("=" * 50)
        
        # Reset environment
        observation, info = self.env.reset()
        
        # Get starting position
        robot_x, robot_y = self.get_robot_position()
        
        # Tracking
        step_count = 0
        frontiers_visited = 0
        start_time = time.time()
        
        # Main exploration loop
        while step_count < self.max_steps:
            
            # Step 1: Update map with robot's sensor
            self.explorer.update_map(robot_x, robot_y, self.sensor_range)
            
            # Step 2: Check progress
            progress = self.explorer.get_exploration_progress() * 100
            
            # Step 3: Find nearest frontier
            frontier = self.explorer.get_nearest_frontier(robot_x, robot_y)
            
            if frontier is None:
                print(f"\n*** EXPLORATION COMPLETE! ***")
                print(f"Total steps: {step_count}")
                print(f"Frontiers visited: {frontiers_visited}")
                print(f"Coverage: {progress:.1f}%")
                break
            
            frontiers_visited += 1
            frontier_x, frontier_y = frontier
            
            print(f"\nFrontier #{frontiers_visited}: ({frontier_x:.1f}, {frontier_y:.1f})")
            print(f"  Progress: {progress:.1f}% explored")
            print(f"  Robot at: ({robot_x:.1f}, {robot_y:.1f})")
            
            # Step 4: Set frontier as goal
            self.env.goal_position = [frontier_x, frontier_y]
            
            # Redraw goal marker
            if self.env.render_mode == "human":
                self.env._draw_goal_marker()
            
            # Step 5: Navigate toward frontier
            for i in range(self.steps_per_frontier):
                # Get action from policy
                action, _ = self.policy.predict(observation, deterministic=True)
                
                # Take action
                observation, reward, done, truncated, info = self.env.step(action)
                step_count += 1
                
                # Update robot position
                robot_x, robot_y = self.get_robot_position()
                
                # Small delay for visualization
                time.sleep(0.02)
                
                # Check if reached frontier (close enough)
                dist_to_frontier = np.sqrt((robot_x - frontier_x)**2 + (robot_y - frontier_y)**2)
                if dist_to_frontier < 0.8:
                    print(f"  Reached frontier in {i+1} steps")
                    break
                
                # Check if hit wall
                if done:
                    print(f"  Hit wall! Resetting...")
                    observation, info = self.env.reset()
                    robot_x, robot_y = self.get_robot_position()
                    break
                
                # Check if timed out
                if truncated:
                    print(f"  Timed out, picking new frontier...")
                    break
        
        # Final stats
        elapsed = time.time() - start_time
        final_progress = self.explorer.get_exploration_progress() * 100
        
        print("\n" + "=" * 50)
        print("EXPLORATION SUMMARY")
        print("=" * 50)
        print(f"Time elapsed: {elapsed:.1f} seconds")
        print(f"Total steps: {step_count}")
        print(f"Frontiers visited: {frontiers_visited}")
        print(f"Final coverage: {final_progress:.1f}%")
        print("=" * 50)
        
        # Show final map
        print("\nFinal map:")
        print(self.explorer.get_map_display())
        
        # Keep window open
        print("\nClose the window to exit.")
        while p.isConnected():
            p.stepSimulation()
            time.sleep(1/240)
        
        self.env.close()


if __name__ == "__main__":
    explorer = AutonomousExplorer()
    explorer.run()
