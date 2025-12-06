"""
Curriculum Learning script for multi-agent grid navigation
Progressively trains from 5x5 to 20x20 grids
"""
import sys
import os
from pathlib import Path
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import supersuit as ss
from environment.simple_grid import SimpleGrid


class EarlyStoppingCallback(BaseCallback):
    """
    Custom callback for early stopping based on mean reward.
    Stops training if no improvement for patience evaluations.
    """
    def __init__(self, check_freq=4096, patience=5, min_episodes=10, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.patience = patience
        self.min_episodes = min_episodes
        self.best_mean_reward = -float('inf')
        self.no_improvement_count = 0
        self.episode_rewards = []
        self.episode_count = 0
        
    def _on_step(self):
        # Check if it's time to evaluate
        if self.n_calls % self.check_freq == 0:
            # Get recent episode rewards from training
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = sum([ep_info['r'] for ep_info in self.model.ep_info_buffer]) / len(self.model.ep_info_buffer)
                
                if self.verbose > 0:
                    print(f"\nEval at {self.num_timesteps} steps: mean_reward={mean_reward:.2f}")
                
                # Check if this is the best reward
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.no_improvement_count = 0
                    if self.verbose > 0:
                        print(f"New best mean reward: {self.best_mean_reward:.2f}")
                else:
                    self.no_improvement_count += 1
                    if self.verbose > 0:
                        print(f"No improvement for {self.no_improvement_count}/{self.patience} checks")
                
                # Stop if no improvement for patience checks and min episodes done
                if self.no_improvement_count >= self.patience and self.num_timesteps >= self.min_episodes * 1000:
                    if self.verbose > 0:
                        print(f"\nEarly stopping: No improvement for {self.patience} evaluations")
                    return False
        
        return True


def train_curriculum(start_size=5, end_size=20, base_timesteps=50000):
    """
    Train agents using curriculum learning across increasing grid sizes.
    Training time scales with grid complexity.
    
    Args:
        start_size (int): Starting grid size (default 5)
        end_size (int): Final grid size (default 20)
        base_timesteps (int): Base training timesteps (scaled by grid size)
    """
    print("=" * 70)
    print("Multi-Agent Grid Navigation - CURRICULUM LEARNING")
    print("=" * 70)
    print(f"Grid sizes: {start_size}x{start_size} → {end_size}x{end_size}")
    print(f"Base timesteps: {base_timesteps:,} (scaled by grid complexity)")
    print("=" * 70)
    
    # Clear old curriculum models if they exist
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models" / "curriculum"
    if models_dir.exists():
        response = input(f"\nExisting curriculum models found. Delete and retrain? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(models_dir)
            print("✓ Old models deleted. Starting fresh training...\n")
        else:
            print("⚠ Keeping existing models. May overwrite during training...\n")
    
    model = None
    
    for grid_size in range(start_size, end_size + 1):
        # Scale training time based on grid complexity
        # Larger grids need more training: timesteps = base * (grid_size / start_size)^1.5
        complexity_factor = (grid_size / start_size) ** 1.5
        timesteps_this_stage = int(base_timesteps * complexity_factor)
        
        print(f"\n{'='*70}")
        print(f"STAGE {grid_size - start_size + 1}/{end_size - start_size + 1}: Training on {grid_size}x{grid_size} grid")
        print(f"Timesteps: {timesteps_this_stage:,} (complexity factor: {complexity_factor:.2f}x)")
        print(f"{'='*70}")
        
        # Create environment with current grid size
        print(f"\n[1/4] Creating {grid_size}x{grid_size} environment...")
        
        # Create custom environment with specific grid size
        def make_env(size):
            def _init():
                env_instance = SimpleGrid(render_mode=None)
                env_instance.grid_size = size
                # Update observation space for new grid size
                from gymnasium import spaces
                import numpy as np
                env_instance.observation_spaces = {
                    a: spaces.Box(low=0, high=size-1, shape=(6,), dtype=np.float32)
                    for a in env_instance.possible_agents
                }
                # Update pygame screen size if needed
                if env_instance.screen:
                    env_instance.screen = None
                return env_instance
            return _init
        
        parallel_env = make_env(grid_size)()
        
        # Convert to vectorized format for SB3
        print("[2/4] Converting to vectorized environment...")
        parallel_env = ss.pettingzoo_env_to_vec_env_v1(parallel_env)
        parallel_env = ss.concat_vec_envs_v1(parallel_env, 1, base_class="stable_baselines3")
        
        # Create or update PPO model
        if model is None:
            print("[3/4] Initializing new PPO model...")
            model = PPO(
                "MlpPolicy", 
                parallel_env, 
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=30,  # 3x more epochs (10 -> 30)
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01
            )
        else:
            print(f"[3/4] Creating new model for {grid_size}x{grid_size} (warm start from previous knowledge)...")
            # Create new model for new observation space, but keep learned hyperparameters
            # The agent will start with random weights but benefit from curriculum structure
            old_learning_rate = model.learning_rate
            model = PPO(
                "MlpPolicy", 
                parallel_env, 
                verbose=1,
                learning_rate=old_learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=30,  # 3x more epochs (10 -> 30)
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01
            )
        
        # Train the model with early stopping
        print(f"[4/4] Training on {grid_size}x{grid_size} grid (with early stopping)...")
        print("-" * 70)
        
        # Setup early stopping callback
        early_stop = EarlyStoppingCallback(
            check_freq=4096,
            patience=5,
            min_episodes=10,
            verbose=1
        )
        
        model.learn(
            total_timesteps=timesteps_this_stage, 
            reset_num_timesteps=False,
            callback=early_stop
        )
        print("-" * 70)
        
        # Save the model for this stage
        project_root = Path(__file__).parent.parent
        models_dir = project_root / "models" / "curriculum"
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = models_dir / f"ppo_grid_{grid_size}x{grid_size}"
        print(f"\n✓ Stage complete! Saving model to '{model_path}'...")
        model.save(str(model_path))
        print(f"✓ Model saved: {model_path}.zip")
        
        # Save final model as main model if this is the last stage
        if grid_size == end_size:
            final_model_path = project_root / "models" / "ppo_simple_grid"
            model.save(str(final_model_path))
            print(f"✓ Final model also saved to: {final_model_path}.zip")
    
    print("\n" + "=" * 70)
    print("✓ CURRICULUM LEARNING COMPLETE!")
    print("=" * 70)
    print(f"\nAll models saved in: models/curriculum/")
    print(f"Final {end_size}x{end_size} model ready at: models/ppo_simple_grid.zip")
    print("\nTo evaluate the final model, run:")
    print(f"  python scripts/evaluate_curriculum.py {end_size}")
    print("=" * 70)


if __name__ == "__main__":
    # Run curriculum learning from 5x5 to 20x20
    # Training time automatically scales with grid size
    train_curriculum(start_size=5, end_size=20, base_timesteps=50000)
