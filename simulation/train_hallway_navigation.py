import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from hallway_nav_env import HallwayNavEnv
import time


def train():
    """Train the hallway navigation policy."""
    
    print("=" * 50)
    print("TABBI Hallway Navigation Training")
    print("=" * 50)
    
    # Create environment (headless for fast training)
    print("\nCreating hallway environment...")
    env = HallwayNavEnv(render_mode=None)
    
    # Check environment
    print("Checking environment...")
    check_env(env, warn=True)
    print("Environment check passed!")
    
    # Create PPO model
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
        device="cpu"
    )
    
    # Train
    print("\n" + "=" * 50)
    print("Starting training for 100,000 timesteps...")
    print("This will take several minutes.")
    print("Watch for 'ep_rew_mean' to increase over time.")
    print("=" * 50 + "\n")
    
    model.learn(total_timesteps=100000)
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    
    # Save model
    model_path = "simulation/hallway_navigation_policy"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    env.close()
    
    return model_path


def test(model_path):
    """Test the trained model."""
    
    print("\n" + "=" * 50)
    print("Testing trained model...")
    print("=" * 50)
    
    # Create environment with GUI
    env = HallwayNavEnv(render_mode="human")
    
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
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            time.sleep(0.02)
        
        if total_reward > 50:
            result = "REACHED GOAL!"
        elif total_reward < -30:
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
    
    import pybullet as p
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1/240)
    
    env.close()


if __name__ == "__main__":
    model_path = train()
    test(model_path)
