"""
Curriculum Learning with MAPPO for multi-agent grid navigation
Progressively trains from 5x5 to 20x20 grids using MAPPO with centralized critic
"""
import sys
import os
from pathlib import Path
import shutil
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from algorithms.mappo import MAPPO
from environment.simple_grid import SimpleGrid
import numpy as np


class EarlyStoppingCallback:
    """
    Early stopping callback for MAPPO
    Stops training if no improvement for patience evaluations.
    """
    def __init__(self, check_freq=4096, patience=5, min_timesteps=10000, verbose=1):
        self.check_freq = check_freq
        self.patience = patience
        self.min_timesteps = min_timesteps
        self.verbose = verbose
        self.best_mean_reward = -float('inf')
        self.no_improvement_count = 0
        self.checks = 0
        self.last_check_timestep = 0
        
    def __call__(self, model: MAPPO) -> bool:
        """Called during rollout collection"""
        # Check at intervals
        if model.total_timesteps - self.last_check_timestep >= self.check_freq:
            self.last_check_timestep = model.total_timesteps
            self.checks += 1
            
            # Get recent episode rewards
            if len(model.episode_rewards) > 0:
                mean_reward = np.mean(model.episode_rewards)
                
                if self.verbose > 0:
                    print(f"\n  [Early Stop Check {self.checks}] Mean reward: {mean_reward:.2f}")
                
                # Check if this is the best reward
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.no_improvement_count = 0
                    if self.verbose > 0:
                        print(f"  ✓ New best: {self.best_mean_reward:.2f}")
                else:
                    self.no_improvement_count += 1
                    if self.verbose > 0:
                        print(f"  ⚠ No improvement: {self.no_improvement_count}/{self.patience}")
                
                # Stop if no improvement and minimum training done
                if self.no_improvement_count >= self.patience and model.total_timesteps >= self.min_timesteps:
                    if self.verbose > 0:
                        print(f"\n  [Early Stopping] Triggered after {self.checks} checks")
                    return False
        
        return True


def train_curriculum(start_size=5, end_size=20, base_timesteps=50000):
    """
    Train agents using curriculum learning with MAPPO across increasing grid sizes.
    Training time scales with grid complexity.
    
    Args:
        start_size (int): Starting grid size (default 5)
        end_size (int): Final grid size (default 20)
        base_timesteps (int): Base training timesteps (scaled by grid size)
    """
    print("=" * 70)
    print("Multi-Agent Grid Navigation - MAPPO CURRICULUM LEARNING")
    print("=" * 70)
    print(f"Grid sizes: {start_size}x{start_size} → {end_size}x{end_size}")
    print(f"Base timesteps: {base_timesteps:,} (scaled by grid complexity)")
    print(f"Algorithm: MAPPO (Centralized Critic)")
    print("=" * 70)
    
    # Create models directory if it doesn't exist
    models_dir = Path(__file__).parent.parent / "models" / "curriculum_mappo"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean old models
    if models_dir.exists():
        print(f"\n[Setup] Cleaning old MAPPO models from {models_dir}")
        for file in models_dir.glob("*.pt"):
            file.unlink()
    
    total_stages = end_size - start_size + 1
    overall_start_time = time.time()
    
    for stage, grid_size in enumerate(range(start_size, end_size + 1), 1):
        print(f"\n{'='*70}")
        print(f"STAGE {stage}/{total_stages}: Training on {grid_size}x{grid_size} grid")
        print(f"{'='*70}")
        
        # Calculate timesteps for this stage (scaled by grid complexity)
        # Formula: base_timesteps * (grid_size / start_size) ^ 1.5
        timesteps = int(base_timesteps * ((grid_size / start_size) ** 1.5))
        print(f"Target timesteps: {timesteps:,}")
        
        # Create environment
        print(f"\n[1/4] Creating {grid_size}x{grid_size} environment...")
        env = SimpleGrid()
        env.grid_size = grid_size
        
        # Create or load model
        model_path = models_dir / f"mappo_grid_{grid_size}x{grid_size}.pt"
        
        print(f"[2/4] Initializing MAPPO...")
        model = MAPPO(
            env=env,
            obs_dim=6,
            action_dim=5,
            num_agents=2,
            lr_actor=3e-4,
            lr_critic=1e-3,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            n_epochs=30,  # 3x standard for better convergence
            batch_size=64,
            buffer_size=2048,
            verbose=1
        )
        
        # Create early stopping callback
        early_stop = EarlyStoppingCallback(
            check_freq=4096,
            patience=5,
            min_timesteps=10000,
            verbose=1
        )
        
        # Train
        print(f"[3/4] Training for up to {timesteps:,} timesteps...")
        print(f"       (Early stopping enabled: patience={early_stop.patience})")
        print("-" * 70)
        
        stage_start_time = time.time()
        model.learn(total_timesteps=timesteps, callback=early_stop)
        stage_duration = time.time() - stage_start_time
        
        # Save model
        print(f"\n[4/4] Saving model...")
        model.save(str(model_path))
        
        # Stage summary
        print(f"\n{'='*70}")
        print(f"STAGE {stage}/{total_stages} COMPLETE")
        print(f"{'='*70}")
        print(f"Grid: {grid_size}x{grid_size}")
        print(f"Timesteps trained: {model.total_timesteps:,}/{timesteps:,}")
        print(f"Updates: {model.num_updates}")
        print(f"Time: {stage_duration:.1f}s ({stage_duration/60:.1f}min)")
        if len(model.episode_rewards) > 0:
            print(f"Final mean reward: {np.mean(model.episode_rewards):.2f}")
        print(f"Model saved: {model_path.name}")
        print(f"{'='*70}")
        
        # Clean up
        env.close()
    
    # Final summary
    total_duration = time.time() - overall_start_time
    print(f"\n{'='*70}")
    print(f"CURRICULUM TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Stages completed: {total_stages}")
    print(f"Grid sizes: {start_size}x{start_size} → {end_size}x{end_size}")
    print(f"Total time: {total_duration:.1f}s ({total_duration/60:.1f}min)")
    print(f"Models saved in: {models_dir}")
    print(f"{'='*70}")
    
    print(f"\nNext steps:")
    print(f"  1. Evaluate models: python scripts/evaluate_mappo.py <grid_size>")
    print(f"  2. Check saved models: {models_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MAPPO Curriculum Training')
    parser.add_argument('--start', type=int, default=5, help='Starting grid size (default: 5)')
    parser.add_argument('--end', type=int, default=20, help='Ending grid size (default: 20)')
    parser.add_argument('--timesteps', type=int, default=50000, help='Base timesteps (default: 50000)')
    
    args = parser.parse_args()
    
    train_curriculum(
        start_size=args.start,
        end_size=args.end,
        base_timesteps=args.timesteps
    )
