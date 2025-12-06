"""
Evaluation script for curriculum-trained models
Visualizes agents on any grid size using Pygame
"""
import sys
import os
from pathlib import Path
import pygame
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from environment.simple_grid import SimpleGrid


def evaluate_curriculum_model(grid_size=20, num_episodes=5):
    """
    Evaluate curriculum-trained PPO agents with Pygame visualization.
    
    Args:
        grid_size (int): Size of the grid to evaluate on
        num_episodes (int): Number of episodes to run
    """
    print("=" * 60)
    print(f"Multi-Agent Grid Navigation - Curriculum Evaluation ({grid_size}x{grid_size})")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    
    # Try to load the specific curriculum model first, then fall back to main model
    model_path = project_root / "models" / "curriculum" / f"ppo_grid_{grid_size}x{grid_size}"
    if not os.path.exists(f"{model_path}.zip"):
        print(f"Curriculum model not found, trying main model...")
        model_path = project_root / "models" / "ppo_simple_grid"
    
    # Check if model exists
    if not os.path.exists(f"{model_path}.zip"):
        print(f"\n❌ Error: Model not found at '{model_path}.zip'")
        print("\nPlease train the model first by running:")
        print("  python scripts/curriculum_train.py")
        return
    
    print(f"Model path: {model_path}")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)
    
    # Load the trained model
    print("\n[1/3] Loading trained model...")
    model = PPO.load(str(model_path))
    
    # Create environment with rendering
    print(f"[2/3] Creating {grid_size}x{grid_size} environment with Pygame rendering...")
    e = SimpleGrid(render_mode="human")
    e.grid_size = grid_size
    
    # Update observation space for the grid size
    from gymnasium import spaces
    import numpy as np
    e.observation_spaces = {
        a: spaces.Box(low=0, high=grid_size-1, shape=(6,), dtype=np.float32)
        for a in e.possible_agents
    }
    
    # Update pygame display
    e.cell = 60
    if e.screen:
        pygame.quit()
    pygame.init()
    e.screen = pygame.display.set_mode((e.cell * grid_size, e.cell * grid_size))
    pygame.display.set_caption(f"Multi-Agent Grid Navigation ({grid_size}x{grid_size})")
    e.clock = pygame.time.Clock()
    
    print("[3/3] Running evaluation...")
    print("\nControls:")
    print("  - Close the Pygame window to stop")
    print("  - Each episode runs until both agents reach their goals")
    print("-" * 60)
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        obs, infos = e.reset()
        
        total_rewards = {agent: 0 for agent in e.agents}
        steps = 0
        done = False
        
        while not done:
            # Handle pygame events to allow window closing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\n\n✓ Window closed. Exiting...")
                    e.close()
                    return
            
            # Get actions for all agents
            actions = {}
            for agent in e.agents:
                if agent in obs:
                    action, _ = model.predict(obs[agent], deterministic=True)
                    actions[agent] = action
            
            # Step the environment
            obs, rewards, terminations, truncations, infos = e.step(actions)
            
            # Track rewards
            for agent in e.agents:
                if agent in rewards:
                    total_rewards[agent] += rewards[agent]
            
            steps += 1
            
            # Check if episode is done (with max steps safety)
            done = all(terminations.values()) or all(truncations.values()) or steps > grid_size * 10
        
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
    
    print("\n" + "=" * 60)
    print("✓ Evaluation complete!")
    print("=" * 60)
    
    # Close environment
    e.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate curriculum-trained agents')
    parser.add_argument('grid_size', type=int, nargs='?', default=20,
                        help='Grid size to evaluate (default: 20)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run (default: 5)')
    
    args = parser.parse_args()
    
    # Evaluate with specified parameters
    evaluate_curriculum_model(grid_size=args.grid_size, num_episodes=args.episodes)
