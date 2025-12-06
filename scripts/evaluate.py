"""
Evaluation script for trained multi-agent grid navigation
Visualizes agents using Pygame
"""
import sys
import os
from pathlib import Path
import pygame

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from environment.simple_grid import SimpleGrid


def evaluate_agents(model_path="models/ppo_simple_grid", num_episodes=5, grid_size=None):
    """
    Evaluate trained PPO agents with Pygame visualization.
    
    Args:
        model_path (str): Path to the trained model
        num_episodes (int): Number of episodes to run
        grid_size (int): Grid size for the environment (auto-detected from path if None)
    """
    print("=" * 60)
    print("Multi-Agent Grid Navigation - Evaluation")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(f"{model_path}.zip"):
        print(f"\n❌ Error: Model not found at '{model_path}.zip'")
        print("\nAvailable curriculum models:")
        curriculum_dir = Path("models/curriculum")
        if curriculum_dir.exists():
            for model_file in sorted(curriculum_dir.glob("*.zip")):
                print(f"  - {model_file.stem}")
        print("\nTo evaluate a curriculum model, run:")
        print("  python scripts/evaluate.py <grid_size>")
        print("Example: python scripts/evaluate.py 6")
        return
    
    print(f"Model path: {model_path}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)
    
    # Load the trained model
    print("\n[1/3] Loading trained model...")
    model = PPO.load(model_path)
    
    # Auto-detect grid size from model path if not specified
    if grid_size is None:
        import re
        match = re.search(r'(\d+)x\d+', model_path)
        if match:
            grid_size = int(match.group(1))
        else:
            grid_size = 5  # Default
    
    # Create environment with rendering
    print(f"[2/3] Creating {grid_size}x{grid_size} environment with Pygame rendering...")
    e = SimpleGrid(render_mode="human")
    e.grid_size = grid_size
    e.reset()
    
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
        max_steps = grid_size * 10  # Same timeout as training
        
        while not done and steps < max_steps:
            # Handle pygame events to allow window closing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\n\n✓ Window closed. Exiting...")
                    e.close()
                    return
            
            # Get actions for all agents (including terminated ones)
            actions = {}
            all_agents = ["player_0", "player_1"]
            for agent in all_agents:
                if agent in obs:
                    action, _ = model.predict(obs[agent], deterministic=True)
                    actions[agent] = action
            
            # Step the environment
            obs, rewards, terminations, truncations, infos = e.step(actions)
            
            # Track rewards
            for agent in all_agents:
                if agent in rewards:
                    total_rewards[agent] += rewards[agent]
            
            steps += 1
            
            # Check if episode is done - ALL agents must be done
            done = all(terminations.values()) or all(truncations.values())
        
        # Print episode summary
        print(f"  Steps: {steps}")
        print(f"  Rewards - Player 0: {total_rewards.get('player_0', 0)}, "
              f"Player 1: {total_rewards.get('player_1', 0)}")
        
        if episode < num_episodes - 1:
            print("  (Starting next episode in 2 seconds...)")
            pygame.time.wait(2000)
    
    print("\n" + "=" * 60)
    print("✓ Evaluation complete!")
    print("=" * 60)
    
    # Close environment
    e.close()


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        grid_size = int(sys.argv[1])
        model_path = f"models/curriculum/ppo_grid_{grid_size}x{grid_size}"
        num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        evaluate_agents(model_path=model_path, num_episodes=num_episodes, grid_size=grid_size)
    else:
        # Try final model first
        if os.path.exists("models/ppo_simple_grid.zip"):
            evaluate_agents(model_path="models/ppo_simple_grid", num_episodes=5, grid_size=20)
        else:
            print("Usage: python scripts/evaluate.py <grid_size> [num_episodes]")
            print("Example: python scripts/evaluate.py 6 5")

