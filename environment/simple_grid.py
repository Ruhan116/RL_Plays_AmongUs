"""
Simple 5x5 Grid Environment for Multi-Agent RL
Two agents navigate to their own goal cells.
"""
import numpy as np
import pygame
from pettingzoo import ParallelEnv
from gymnasium import spaces


class SimpleGrid(ParallelEnv):
    """
    A simple 5x5 grid environment where 2 agents try to reach their own goal cells.
    
    Agents: player_0, player_1
    Grid Size: 5x5
    Actions: 0=up, 1=down, 2=left, 3=right, 4=stay
    Reward: +1 for reaching goal, 0 otherwise
    Episode: Ends when both agents reach their goals
    """
    
    metadata = {
        "render_modes": ["human"], 
        "name": "simple_grid",
    }

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.grid_size = 5

        # Define agents
        self.agents = ["player_0", "player_1"]
        self.possible_agents = self.agents[:]
        
        # Agent positions and goals (will be set in reset)
        self.pos = {}
        self.goals = {}
        self._np_random = None
        
        # Episode step tracking
        self.current_step = 0
        self.max_steps = self.grid_size * 10  # 10x grid size

        # Define action and observation spaces
        # Observation: [agent_x, agent_y, goal_x, goal_y, other_agent_x, other_agent_y]
        self.action_spaces = {a: spaces.Discrete(5) for a in self.possible_agents}
        self.observation_spaces = {
            a: spaces.Box(low=0, high=self.grid_size-1, shape=(6,), dtype=np.float32)
            for a in self.possible_agents
        }

        # Pygame rendering variables
        self.screen = None
        self.cell = 60
        self.clock = None
        
        # Initialize pygame if rendering is enabled
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.cell * 5, self.cell * 5))
            pygame.display.set_caption("Multi-Agent Grid Navigation")
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state with randomized goals."""
        if seed is not None:
            self._np_random = np.random.RandomState(seed)
        elif self._np_random is None:
            self._np_random = np.random.RandomState()
        
        # Randomly select 4 unique positions for 2 agents and 2 goals
        all_positions = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        selected_positions = self._np_random.choice(len(all_positions), size=4, replace=False)
        positions = [all_positions[i] for i in selected_positions]
        
        # Assign positions
        self.pos = {
            "player_0": positions[0],
            "player_1": positions[1]
        }
        
        self.goals = {
            "player_0": positions[2],
            "player_1": positions[3]
        }
        
        # Reset agent states
        self.agents = self.possible_agents[:]
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        
        # Reset step counter and update max steps based on current grid size
        self.current_step = 0
        self.max_steps = self.grid_size * 10
        self.current_step = 0
        
        # Update max steps based on grid size
        self.max_steps = self.grid_size * 10

        # Return observations for all agents
        observations = {agent: self.observe(agent) for agent in self.agents}
        return observations, self.infos

    def observe(self, agent):
        """
        Generate observation for the given agent.
        Returns [agent_x, agent_y, goal_x, goal_y, other_agent_x, other_agent_y]
        """
        if agent not in self.pos:
            return np.zeros(6, dtype=np.float32)
        
        # Get agent's position and goal
        ax, ay = self.pos[agent]
        gx, gy = self.goals[agent]
        
        # Get other agent's position
        other_agent = "player_1" if agent == "player_0" else "player_0"
        ox, oy = self.pos.get(other_agent, (0, 0))
        
        return np.array([ax, ay, gx, gy, ox, oy], dtype=np.float32)

    def step(self, actions):
        """
        Execute one step in the environment for all agents.
        
        Args:
            actions: Dictionary mapping agent names to actions
            
        Returns:
            observations, rewards, terminations, truncations, infos
        """
        self.current_step += 1
        # Execute actions for all agents
        for agent, action in actions.items():
            if agent not in self.agents:
                continue
                
            x, y = self.pos[agent]
            
            # Apply action: 0=up, 1=down, 2=left, 3=right, 4=stay
            if action == 0:  # up
                x = max(0, x - 1)
            elif action == 1:  # down
                x = min(self.grid_size - 1, x + 1)
            elif action == 2:  # left
                y = max(0, y - 1)
            elif action == 3:  # right
                y = min(self.grid_size - 1, y + 1)
            # action == 4: stay still
            
            self.pos[agent] = (x, y)

        # Calculate rewards and check terminations
        rewards = {}
        for agent in self.agents:
            # Reward structure:
            # - First time reaching goal: 1.0 (full reward)
            # - Staying at goal after reaching it: 0.5 (half reward to encourage staying)
            # - Not at goal: 0.0
            if self.pos[agent] == self.goals[agent]:
                if not self.terminations[agent]:
                    # First time reaching goal
                    rewards[agent] = 1.0
                    self.terminations[agent] = True
                else:
                    # Already reached goal, give continuous reward for staying
                    rewards[agent] = 0.5
            else:
                rewards[agent] = 0.0

        # Episode ends when all agents reach their goals OR timeout
        if all(self.terminations.values()):
            self.truncations = {a: True for a in self.agents}
        
        # Truncate episode if it takes too long (prevents infinite wandering)
        if self.current_step >= self.max_steps:
            self.truncations = {a: True for a in self.agents}

        # Get observations for all agents

        # Get observations for all agents
        observations = {agent: self.observe(agent) for agent in self.agents}

        # Render if in human mode
        if self.render_mode == "human":
            self.render()

        return observations, rewards, self.terminations, self.truncations, self.infos

    def render(self):
        """Render the environment using Pygame."""
        if self.screen is None:
            return
            
        # Fill background
        self.screen.fill((0, 0, 0))
        
        # Draw grid lines
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pygame.draw.rect(
                    self.screen,
                    (40, 40, 40),
                    (j * self.cell, i * self.cell, self.cell, self.cell),
                    1
                )

        # Draw goal cells
        for agent, (x, y) in self.goals.items():
            color = (0, 200, 0) if agent == "player_0" else (200, 0, 0)
            pygame.draw.rect(
                self.screen, 
                color,
                (y * self.cell, x * self.cell, self.cell, self.cell)
            )
            
            # Add label
            font = pygame.font.Font(None, 24)
            text = font.render("G0" if agent == "player_0" else "G1", True, (255, 255, 255))
            text_rect = text.get_rect(center=(y * self.cell + self.cell // 2, 
                                             x * self.cell + self.cell // 2))
            self.screen.blit(text, text_rect)

        # Draw agents
        for agent, (x, y) in self.pos.items():
            color = (0, 100, 255) if agent == "player_0" else (255, 200, 0)
            pygame.draw.circle(
                self.screen, 
                color,
                (y * self.cell + self.cell // 2, x * self.cell + self.cell // 2),
                self.cell // 3
            )
            
            # Add label
            font = pygame.font.Font(None, 24)
            text = font.render("A0" if agent == "player_0" else "A1", True, (255, 255, 255))
            text_rect = text.get_rect(center=(y * self.cell + self.cell // 2, 
                                             x * self.cell + self.cell // 2))
            self.screen.blit(text, text_rect)

        pygame.display.flip()
        if self.clock:
            self.clock.tick(10)  # 10 FPS

    def close(self):
        """Clean up resources."""
        if self.screen:
            pygame.quit()
            self.screen = None
            self.clock = None


def env(render_mode=None):
    """Factory function to create environment instance."""
    return SimpleGrid(render_mode)
