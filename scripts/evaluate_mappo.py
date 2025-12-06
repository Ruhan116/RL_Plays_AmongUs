"""
Evaluate MAPPO-trained agents with visualization

Usage:
    python scripts/evaluate_mappo.py 10              # Evaluate 10x10 model
    python scripts/evaluate_mappo.py 20 --episodes 10  # Run 10 episodes on 20x20
"""

import numpy as np
import pygame
import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.simple_grid import SimpleGrid
from algorithms.mappo import MAPPO
import torch


def evaluate_mappo_model(grid_size=20, num_episodes=5):
    """
    Evaluate a trained MAPPO model.
    
    Args:
        grid_size: Grid size to evaluate
        num_episodes: Number of episodes to run
    """
    print("="*60)
    print(f"Multi-Agent Grid Navigation - MAPPO Evaluation ({grid_size}x{grid_size})")
    print("="*60)
    
    # Model path
    model_path = f"models/mappo_curriculum/mappo_grid_{grid_size}x{grid_size}.pt"
    print(f"Model path: {os.path.abspath(model_path)}")
    
    if not os.path.exists(model_path):
        print(f"\n✗ Error: Model file not found!")
        print(f"  Expected: {model_path}")
        print(f"\n  Make sure you've trained the model first:")
        print(f"    python scripts/train_mappo_curriculum.py")
        return
    
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Episodes: {num_episodes}")
    print("="*60)
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[1/3] Using device: {device}")
    if device == "cuda":
        print(f"      GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    print(f"[2/3] Loading trained MAPPO model...")
    model = MAPPO(
        obs_dim=6,
        action_dim=5,
        num_agents=2,
        device=device
    )
    model.load(model_path)
    print("      ✓ Model loaded successfully")
    
    # Create environment with rendering
    print(f"[3/3] Creating {grid_size}x{grid_size} environment with Pygame rendering...")
    env = SimpleGrid(render_mode="human")
    env.grid_size = grid_size
    print("      ✓ Environment created")
    
    print("\n" + "-"*60)
    print("\nControls:")
    print("  - Close the Pygame window to stop")
    print("  - Each episode runs until both agents reach their goals")
    print("-"*60)
    
    # Run evaluation episodes
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        obs, _ = env.reset()
        
        total_rewards = {agent: 0 for agent in env.agents}
        steps = 0
        done = False
        
        while not done:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\n\n✓ Window closed. Exiting...")
                    env.close()
                    return
            
            # Get actions from MAPPO model (deterministic for evaluation)
            actions, _, _ = model.select_actions(obs, deterministic=True)
            
            # Step environment
            obs, rewards, terminations, truncations, _ = env.step(actions)
            
            # Track rewards
            for agent in env.agents:
                if agent in rewards:
                    total_rewards[agent] += rewards[agent]
            
            steps += 1
            
            # Check if done (with max steps safety)
            done = all(terminations.values()) or all(truncations.values()) or steps > grid_size * 10
            
            # Small delay for visualization
            pygame.time.wait(50)
        
        # Print episode summary
        print(f"  Steps: {steps}")
        print(f"  Rewards - Player 0: {total_rewards.get('player_0', 0)}, "
              f"Player 1: {total_rewards.get('player_1', 0)}")
        
        if total_rewards.get('player_0', 0) == 1 and total_rewards.get('player_1', 0) == 1:
            print(f"  ✓ Both agents reached their goals!")
        else:
            print(f"  ⚠ Some agents didn't reach their goals")
        
        if episode < num_episodes - 1:
            print("  (Starting next episode in 2 seconds...)")
            pygame.time.wait(2000)
    
    print("\n" + "="*60)
    print("✓ Evaluation complete!")
    print("="*60)
    
    # Close environment
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate MAPPO-trained agents')
    parser.add_argument('grid_size', type=int, nargs='?', default=20,
                        help='Grid size to evaluate (default: 20)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run (default: 5)')
    
    args = parser.parse_args()
    
    # Evaluate with specified parameters
    evaluate_mappo_model(grid_size=args.grid_size, num_episodes=args.episodes)
