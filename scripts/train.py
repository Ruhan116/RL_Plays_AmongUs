"""
Training script for multi-agent grid navigation using PPO
"""
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
import supersuit as ss
from environment.simple_grid import env


def train_agents(total_timesteps=50000, model_path="models/ppo_simple_grid"):
    """
    Train PPO agents on the multi-agent grid environment.
    
    Args:
        total_timesteps (int): Total number of training timesteps
        model_path (str): Path to save the trained model
    """
    print("=" * 60)
    print("Multi-Agent Grid Navigation - PPO Training")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Model save path: {model_path}")
    print("=" * 60)
    
    # Create the environment
    print("\n[1/3] Creating parallel environment...")
    parallel_env = env()
    
    # Convert to vectorized format for SB3
    print("[2/3] Converting to vectorized environment...")
    parallel_env = ss.pettingzoo_env_to_vec_env_v1(parallel_env)
    parallel_env = ss.concat_vec_envs_v1(parallel_env, 1, base_class="stable_baselines3")
    
    # Create PPO model
    print("[3/3] Initializing PPO model...")
    model = PPO(
        "MlpPolicy", 
        parallel_env, 
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01
    )
    
    # Train the model
    print("[4/4] Training agents...")
    print("-" * 60)
    model.learn(total_timesteps=total_timesteps)
    print("-" * 60)
    
    # Save the trained model
    print(f"\n✓ Training complete! Saving model to '{model_path}'...")
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    
    print(f"✓ Model saved successfully!")
    print("\nTo evaluate the trained agents, run:")
    print(f"  python scripts/evaluate.py")
    print("=" * 60)


if __name__ == "__main__":
    # Get project root directory
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "ppo_simple_grid"
    
    # Train with default parameters
    train_agents(total_timesteps=50000, model_path=str(model_path))
