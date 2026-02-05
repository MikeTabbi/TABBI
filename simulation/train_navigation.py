import sys
import os

# Add simulation folder to path so we can import our environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from navigation_env import NavigationEnv
import time


def train():
    """Train the navigation policy."""
    
    print("=" * 50)
    print("TABBI Navigation Policy Training")
    print("=" * 50)
    
    # Step 1: Create environment (headless for fast training)
    print("\nCreating environment...")
    env = NavigationEnv(render_mode=None)
    
    # Step 2: Check environment is valid
    print("Checking environment...")
    check_env(env, warn=True)
    print("Environment check passed!")
    
    # Step 3: Create PPO model
    print("\nCreating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        device="cpu"  # Use CPU (MPS can have issues with SB3)
    )
    
    # Step 4: Train
    print("\n" + "=" * 50)
    print("Starting training for 50,000 timesteps...")
    print("This will take a few minutes.")
    print("Watch for 'ep_rew_mean' to increase over time.")
    print("=" * 50 + "\n")
    
    model.learn(total_timesteps=50000)
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    
    # Step 5: Save model
    model_path = "simulation/navigation_policy"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Close training environment
    env.close()
    
    return model_path


def test(model_path):
    """Test the trained model."""
    
    print("\n" + "=" * 50)
    print("Testing trained model...")
    print("=" * 50)
    
    # Create environment with GUI so we can watch
    env = NavigationEnv(render_mode="human")
    
    # Load trained model
    model = PPO.load(model_path)
    
    # Run 5 test episodes
    for episode in range(1, 6):
        observation, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode}:")
        print(f"  Goal position: {env.goal_position}")
        
        while not done and not truncated:
            # AI chooses action
            action, _ = model.predict(observation, deterministic=True)
            
            # Take action
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Small delay so we can watch
            time.sleep(0.02)
        
        # Report results
        if total_reward > 50:
            result = "REACHED GOAL!"
        elif total_reward < -50:
            result = "Hit wall"
        else:
            result = "Timed out"
        
        print(f"  Result: {result}")
        print(f"  Steps: {steps}")
        print(f"  Total reward: {total_reward:.1f}")
    
    print("\n" + "=" * 50)
    print("Testing complete!")
    print("Close the window to exit.")
    print("=" * 50)
    
    # Keep window open
    import pybullet as p
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1/240)
    
    env.close()


if __name__ == "__main__":
    # Train the model
    model_path = train()
    
    # Test the trained model
    test(model_path)