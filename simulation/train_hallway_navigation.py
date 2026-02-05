import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from straight_hallway_env import StraightHallwayEnv  # CHANGED: Use simple straight hallway
import time


def train():
    """Train the hallway navigation policy with TensorBoard monitoring."""
    
    # Define directories for saving
    checkpoint_dir = "./checkpoints/"
    tensorboard_dir = "./tensorboard_logs/"
    
    print("=" * 50)
    print("TABBI Hallway Navigation Training")
    print("SIMPLE STRAIGHT HALLWAY")
    print("WITH TENSORBOARD MONITORING")
    print("=" * 50)
    print(f"\nCheckpoints will be saved to: {checkpoint_dir}")
    print(f"TensorBoard logs will be saved to: {tensorboard_dir}")
    print("\nTo view training progress in real-time:")
    print("  1. Open a new terminal")
    print("  2. Run: tensorboard --logdir=./tensorboard_logs")
    print("  3. Open browser to: http://localhost:6006")
    print("=" * 50)
    
    # Create environment (headless for fast training)
    print("\nCreating SIMPLE STRAIGHT hallway environment...")
    env = StraightHallwayEnv(render_mode=None)  # CHANGED: Simple hallway
    
    # Check environment
    print("Checking environment...")
    check_env(env, warn=True)
    print("Environment check passed!")
    
    # Create checkpoint callback
    print("\nSetting up checkpoint system...")
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,  # Save every 100k steps
        save_path=checkpoint_dir,
        name_prefix="straight_hallway_nav",  # CHANGED: New prefix
        verbose=1
    )
    print("Checkpoints will be saved every 100,000 steps")
    
    # Create PPO model with TensorBoard logging
    print("\nCreating PPO model with TensorBoard logging...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=tensorboard_dir,  # Enable TensorBoard
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        device="cpu"
    )
    
    # Train
    print("\n" + "=" * 50)
    print("Starting training for 500,000 timesteps...")
    print("Environment: SIMPLE STRAIGHT HALLWAY (12m x 3m)")
    print("This will take approximately 30-60 minutes.")
    print("The robot is learning in headless mode (no GUI).")
    print("You'll see the trained robot perform when testing!")
    print("Watch for 'ep_rew_mean' to increase over time.")
    print("Checkpoints: 100k, 200k, 300k, 400k, 500k steps")
    print("=" * 50 + "\n")
    
    model.learn(
        total_timesteps=500000,  # Full training
        callback=checkpoint_callback,
        tb_log_name="straight_hallway_run"  # CHANGED: New run name
    )
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    
    # Save final model
    model_path = "straight_hallway_navigation_policy_final"  # CHANGED: New name
    model.save(model_path)
    print(f"\nFinal model saved to: {model_path}")
    
    print("\n" + "=" * 50)
    print("TRAINING ANALYSIS")
    print("=" * 50)
    print("\nTo view training graphs:")
    print("  tensorboard --logdir=./tensorboard_logs")
    print("  Then open: http://localhost:6006")
    print("\nCheckpoints saved at:")
    print(f"  {checkpoint_dir}")
    print("=" * 50)
    
    env.close()
    
    return model_path


def test(model_path):
    """Test the trained model."""
    
    print("\n" + "=" * 50)
    print("Testing trained model...")
    print("=" * 50)
    
    # Create environment with GUI
    env = StraightHallwayEnv(render_mode="human")  # CHANGED: Simple hallway
    
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
    
    # Ask user if they want to test
    print("\n" + "=" * 50)
    response = input("Do you want to test the trained model now? (y/n): ")
    if response.lower() == 'y':
        test(model_path)
    else:
        print("Skipping testing. You can test later by running:")
        print(f"  python -c 'from train_hallway_navigation import test; test(\"{model_path}\")'")