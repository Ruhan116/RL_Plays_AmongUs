"""
Curriculum Training with MAPPO

Trains agents progressively from 5x5 to 20x20 grids using MAPPO with:
- Centralized critic for better value estimation
- Decentralized actors for scalable execution
- GPU acceleration
- Value normalization
- Learning rate scheduling
- Early stopping based on success rate
"""

import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.simple_grid import SimpleGrid
from algorithms.mappo import MAPPO
import torch


class RolloutBuffer:
    """Buffer to collect rollout data for MAPPO training."""
    def __init__(self, num_agents):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.num_agents = num_agents
    
    def add(self, obs, actions, log_probs, rewards, value, done):
        """Add one timestep of data."""
        self.observations.append(obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.values.append(value)
        self.dones.append(done)
    
    def get(self, next_value):
        """Get all data as numpy arrays."""
        return {
            'observations': np.array(self.observations),  # [T, num_agents, obs_dim]
            'actions': np.array(self.actions),  # [T, num_agents]
            'log_probs': np.array(self.log_probs),  # [T, num_agents]
            'rewards': np.array(self.rewards),  # [T, num_agents]
            'values': np.array(self.values),  # [T]
            'dones': np.array(self.dones),  # [T]
            'next_value': next_value
        }
    
    def clear(self):
        """Clear buffer."""
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []


def collect_rollouts(env, model, n_steps, buffer):
    """
    Collect rollouts from environment.
    
    Args:
        env: Environment instance
        model: MAPPO model
        n_steps: Number of steps to collect
        buffer: RolloutBuffer to store data
    Returns:
        mean_reward: Average reward per episode
        success_rate: Fraction of episodes where both agents reached goals
    """
    obs, _ = env.reset()
    episode_rewards = []
    episode_reward = {agent: 0 for agent in env.agents}
    successes = []
    episode_steps = 0
    max_episode_steps = env.grid_size * 10  # Limit episode length
    
    for step in range(n_steps):
        # Get actions from model
        actions, log_probs, value = model.select_actions(obs, deterministic=False)
        
        # Step environment
        next_obs, rewards, terminations, truncations, _ = env.step(actions)
        
        # Convert to arrays for buffer
        obs_array = np.array([obs[agent] for agent in env.agents])
        actions_array = np.array([actions[agent] for agent in env.agents])
        log_probs_array = np.array([log_probs[agent] for agent in env.agents])
        rewards_array = np.array([rewards[agent] for agent in env.agents])
        
        # Check if episode done (include step limit)
        episode_steps += 1
        done = all(terminations.values()) or all(truncations.values()) or episode_steps >= max_episode_steps
        
        # Store in buffer
        buffer.add(obs_array, actions_array, log_probs_array, rewards_array, value, done)
        
        # Track episode rewards
        for agent in env.agents:
            episode_reward[agent] += rewards[agent]
        
        # Reset if done
        if done:
            # Check if both agents succeeded (terminations, not truncations or timeout)
            success = all(terminations.values()) and episode_steps < max_episode_steps
            successes.append(1.0 if success else 0.0)
            
            # Store episode reward
            mean_episode_reward = np.mean([episode_reward[a] for a in env.agents])
            episode_rewards.append(mean_episode_reward)
            
            # Reset
            obs, _ = env.reset()
            episode_reward = {agent: 0 for agent in env.agents}
            episode_steps = 0
        else:
            obs = next_obs
    
    # Compute final value for last episode
    if not done:
        _, _, next_value = model.select_actions(obs, deterministic=False)
    else:
        next_value = 0.0
    
    mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
    success_rate = np.mean(successes) if successes else 0.0
    
    return mean_reward, success_rate, next_value


def train_grid_size(grid_size, model=None, device="cuda"):
    """
    Train agents on a specific grid size using MAPPO.
    
    Args:
        grid_size: Size of the grid (grid_size x grid_size)
        model: Existing MAPPO model to continue training (None for new model)
        device: Device to use ('cuda' or 'cpu')
    Returns:
        model: Trained MAPPO model
        final_success_rate: Success rate on last evaluation
    """
    print(f"\n{'='*60}")
    print(f"Training on {grid_size}x{grid_size} grid")
    print(f"{'='*60}")
    
    # Create environment
    env = SimpleGrid()
    env.grid_size = grid_size
    env.reset()
    
    # Calculate training timesteps (scaled by grid complexity)
    base_timesteps = 200000  # Increased from 50k to 200k
    timesteps = int(base_timesteps * (grid_size / 5) ** 1.5)
    n_steps = 2048  # Steps per rollout
    n_rollouts = timesteps // n_steps
    
    # Scale epochs with grid size for more training on complex grids
    base_epochs = 30
    n_epochs = min(50, int(base_epochs * (grid_size / 5) ** 0.5))
    
    print(f"Total timesteps: {timesteps:,}")
    print(f"Steps per rollout: {n_steps}")
    print(f"Number of rollouts: {n_rollouts}")
    print(f"Epochs per update: {n_epochs}")
    
    # Create or reuse MAPPO model
    if model is None:
        print(f"Creating new MAPPO model on {device}")
        model = MAPPO(
            obs_dim=6,
            action_dim=5,
            num_agents=2,
            device=device,
            lr_actor=3e-4,
            lr_critic=1e-3,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            max_grad_norm=0.5,
            n_epochs=n_epochs,
            batch_size=64,
            normalize_advantage=True,
            normalize_value=True
        )
    else:
        print(f"Continuing training with existing MAPPO model")
        # Update n_epochs for current grid size
        model.n_epochs = n_epochs
        # Reset learning rate schedulers to prevent excessive decay
        for scheduler in model.actor_schedulers:
            scheduler.base_lrs = [3e-4]
            scheduler.last_epoch = -1
        model.critic_scheduler.base_lrs = [1e-3]
        model.critic_scheduler.last_epoch = -1
    
    # Create rollout buffer
    buffer = RolloutBuffer(num_agents=2)
    
    # Training loop
    best_success_rate = 0.0
    no_improvement = 0
    patience = 5
    
    for rollout_idx in range(n_rollouts):
        # Collect rollouts
        buffer.clear()
        mean_reward, success_rate, next_value = collect_rollouts(env, model, n_steps, buffer)
        
        # Update model
        rollout_data = buffer.get(next_value)
        train_info = model.update(rollout_data)
        
        # Print progress
        if rollout_idx % 5 == 0 or rollout_idx == n_rollouts - 1:
            print(f"Rollout {rollout_idx+1}/{n_rollouts} | "
                  f"Reward: {mean_reward:.3f} | "
                  f"Success: {success_rate:.1%} | "
                  f"Actor Loss: {train_info['actor_loss']:.4f} | "
                  f"Critic Loss: {train_info['critic_loss']:.4f} | "
                  f"Entropy: {train_info['entropy']:.4f} | "
                  f"KL: {train_info['approx_kl']:.4f} | "
                  f"LR: {train_info['actor_lr']:.2e}")
        
        # Track best success rate
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            no_improvement = 0
        else:
            no_improvement += 1
        
        # Only stop if consistently achieving near-perfect success and past minimum training
        if success_rate >= 0.95 and rollout_idx >= n_rollouts // 4:
            print(f"\n✓ Converged! Success rate: {success_rate:.1%}")
            break
    
    print(f"\nFinal success rate: {best_success_rate:.1%}")
    
    return model, best_success_rate


def main():
    """Main curriculum training loop."""
    print("="*60)
    print("MAPPO Curriculum Training")
    print("Multi-Agent Navigation with Centralized Critic")
    print("="*60)
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Create models directory
    models_dir = "models/mappo_curriculum"
    os.makedirs(models_dir, exist_ok=True)
    
    # Curriculum schedule
    grid_sizes = list(range(5, 21))  # 5x5 to 20x20
    print(f"\nCurriculum: {len(grid_sizes)} stages from {grid_sizes[0]}x{grid_sizes[0]} to {grid_sizes[-1]}x{grid_sizes[-1]}")
    
    # Train on each grid size
    model = None
    results = []
    
    for stage_idx, grid_size in enumerate(grid_sizes):
        print(f"\n{'#'*60}")
        print(f"# Stage {stage_idx + 1}/{len(grid_sizes)}: {grid_size}x{grid_size} Grid")
        print(f"{'#'*60}")
        
        # Train
        model, success_rate = train_grid_size(grid_size, model, device)
        
        # Save model
        model_path = os.path.join(models_dir, f"mappo_grid_{grid_size}x{grid_size}.pt")
        model.save(model_path)
        print(f"✓ Model saved: {model_path}")
        
        # Track results
        results.append({
            'grid_size': grid_size,
            'success_rate': success_rate
        })
        
        # Evaluate every 5 stages
        if (stage_idx + 1) % 5 == 0 or stage_idx == len(grid_sizes) - 1:
            print(f"\n{'*'*60}")
            print(f"* EVALUATION CHECKPOINT - Stage {stage_idx + 1}/{len(grid_sizes)}")
            print(f"{'*'*60}")
            print(f"\nRunning quick evaluation on {grid_size}x{grid_size}...")
            
            # Run 3 test episodes
            env = SimpleGrid()
            env.grid_size = grid_size
            successes = []
            
            for test_ep in range(3):
                obs, _ = env.reset()
                done = False
                steps = 0
                
                while not done and steps < grid_size * 10:
                    actions, _, _ = model.select_actions(obs, deterministic=True)
                    obs, rewards, terminations, truncations, _ = env.step(actions)
                    done = all(terminations.values()) or all(truncations.values())
                    steps += 1
                
                success = all(terminations.values())
                successes.append(success)
                print(f"  Episode {test_ep+1}/3: {'✓ Success' if success else '✗ Failed'} ({steps} steps)")
            
            eval_success_rate = np.mean(successes)
            print(f"\nEvaluation Success Rate: {eval_success_rate:.1%}")
            print(f"{'*'*60}\n")
    
    # Print summary
    print("\n" + "="*60)
    print("CURRICULUM TRAINING COMPLETE!")
    print("="*60)
    print("\nResults Summary:")
    print(f"{'Grid Size':<12} {'Success Rate':<15}")
    print("-" * 30)
    for r in results:
        print(f"{r['grid_size']}x{r['grid_size']:<8} {r['success_rate']:.1%}")
    
    print(f"\n✓ All models saved in: {models_dir}")
    print(f"✓ Final model: mappo_grid_20x20.pt")
    print("\nTo evaluate, run:")
    print("  python scripts/evaluate_mappo.py 20")


if __name__ == "__main__":
    main()
