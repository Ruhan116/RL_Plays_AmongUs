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
from environment.simple_grid import env


def evaluate_agents(model_path="models/ppo_simple_grid", num_episodes=5):
    """
    Evaluate trained PPO agents with Pygame visualization.
    
    Args:
        model_path (str): Path to the trained model
        num_episodes (int): Number of episodes to run
    """
    print("=" * 60)
    print("Multi-Agent Grid Navigation - Evaluation")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(f"{model_path}.zip"):
        print(f"\n❌ Error: Model not found at '{model_path}.zip'")
        print("\nPlease train the model first by running:")
        print("  python scripts/train.py")
        return
    
    print(f"Model path: {model_path}")
    print(f"Episodes: {num_episodes}")
    print("=" * 60)
    
    # Load the trained model
    print("\n[1/3] Loading trained model...")
    model = PPO.load(model_path)
    
    # Create environment with rendering
    print("[2/3] Creating environment with Pygame rendering...")
    e = env(render_mode="human")
    
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
            
            # Check if episode is done
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
    # Get project root directory
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "ppo_simple_grid"
    
    # Evaluate with default parameters
    evaluate_agents(model_path=str(model_path), num_episodes=5)
