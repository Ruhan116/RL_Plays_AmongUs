"""
Multi-Agent Proximal Policy Optimization (MAPPO) Implementation

This module implements MAPPO with:
- Centralized value function (critic sees global state)
- Decentralized policy (actor sees local observations)
- Value normalization for stability
- Gradient clipping
- Learning rate scheduling
- GPU acceleration support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple
import os


class CentralizedCritic(nn.Module):
    """
    Centralized critic that sees global state (all agents' observations).
    Uses this global information to better estimate value function.
    """
    def __init__(self, global_state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Normalization for stability
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights with orthogonal initialization
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization for better training stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, global_state):
        """
        Args:
            global_state: [batch, global_state_dim] - concatenated observations of all agents
        Returns:
            value: [batch, 1] - estimated value of the global state
        """
        return self.net(global_state)


class DecentralizedActor(nn.Module):
    """
    Decentralized actor that only sees local observations.
    Each agent can have its own actor or share parameters.
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)  # Smaller gain for policy
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, obs):
        """
        Args:
            obs: [batch, obs_dim] - local observation
        Returns:
            logits: [batch, action_dim] - action logits
        """
        return self.net(obs)
    
    def get_action(self, obs, deterministic=False):
        """
        Sample action from policy.
        
        Args:
            obs: [batch, obs_dim] or [obs_dim]
            deterministic: If True, take argmax; if False, sample
        Returns:
            action: sampled action
            log_prob: log probability of action
            entropy: entropy of action distribution
        """
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)
        
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy


class RunningMeanStd:
    """
    Running mean and std for value normalization.
    Tracks statistics across training for stable value predictions.
    """
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
    
    def update(self, x):
        """Update running statistics."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update from batch statistics."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    def normalize(self, x):
        """Normalize values."""
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class MAPPO:
    """
    Multi-Agent PPO with centralized critic and decentralized actors.
    
    Features:
    - Centralized value function for better credit assignment
    - Decentralized policies for scalability
    - Value normalization for training stability
    - Gradient clipping to prevent large updates
    - Learning rate scheduling
    - GPU acceleration
    """
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_agents: int,
        device: str = "cuda",
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 30,
        batch_size: int = 64,
        normalize_advantage: bool = True,
        normalize_value: bool = True
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.normalize_advantage = normalize_advantage
        self.normalize_value = normalize_value
        
        # Global state dimension = all agents' observations concatenated
        global_state_dim = obs_dim * num_agents
        
        # Create networks
        self.actors = nn.ModuleList([
            DecentralizedActor(obs_dim, action_dim).to(self.device)
            for _ in range(num_agents)
        ])
        self.critic = CentralizedCritic(global_state_dim).to(self.device)
        
        # Optimizers with weight decay for regularization
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=lr_actor, eps=1e-5, weight_decay=1e-5)
            for actor in self.actors
        ]
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=lr_critic, eps=1e-5, weight_decay=1e-5
        )
        
        # Learning rate schedulers (cosine annealing)
        self.actor_schedulers = [
            optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000, eta_min=1e-5)
            for opt in self.actor_optimizers
        ]
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=1000, eta_min=1e-5
        )
        
        # Value normalization
        if normalize_value:
            self.value_normalizer = RunningMeanStd(shape=())
        
        # Training statistics
        self.update_count = 0
        
    def select_actions(self, observations: Dict[str, np.ndarray], deterministic: bool = False):
        """
        Select actions for all agents.
        
        Args:
            observations: Dict mapping agent_id to observation array
            deterministic: Whether to use deterministic policy
        Returns:
            actions: Dict mapping agent_id to action
            log_probs: Dict mapping agent_id to log probability
            values: Centralized value estimate
        """
        self.actors[0].eval()
        self.actors[1].eval()
        self.critic.eval()
        
        with torch.no_grad():
            # Process each agent's observation
            obs_tensors = []
            actions = {}
            log_probs = {}
            
            for i, (agent_id, obs) in enumerate(observations.items()):
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                obs_tensors.append(obs_tensor)
                
                action, log_prob, _ = self.actors[i].get_action(obs_tensor, deterministic)
                actions[agent_id] = action.cpu().item()
                log_probs[agent_id] = log_prob.cpu().item()
            
            # Compute centralized value
            global_state = torch.cat(obs_tensors, dim=-1)
            value = self.critic(global_state).cpu().item()
        
        self.actors[0].train()
        self.actors[1].train()
        self.critic.train()
        
        return actions, log_probs, value
    
    def compute_gae(self, rewards, values, dones, next_value):
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: [T] array of rewards
            values: [T] array of value estimates
            dones: [T] array of done flags
            next_value: Value estimate for the next state
        Returns:
            advantages: [T] array of advantages
            returns: [T] array of returns
        """
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, rollout_buffer):
        """
        Update policy and value function using rollout data.
        
        Args:
            rollout_buffer: Dictionary containing rollout data
        Returns:
            train_info: Dictionary of training statistics
        """
        # Extract rollout data
        observations = rollout_buffer['observations']  # [T, num_agents, obs_dim]
        actions = rollout_buffer['actions']  # [T, num_agents]
        old_log_probs = rollout_buffer['log_probs']  # [T, num_agents]
        rewards = rollout_buffer['rewards']  # [T, num_agents]
        values = rollout_buffer['values']  # [T]
        dones = rollout_buffer['dones']  # [T]
        next_value = rollout_buffer['next_value']
        
        # Compute advantages using mean reward across agents
        mean_rewards = np.mean(rewards, axis=1)
        advantages, returns = self.compute_gae(mean_rewards, values, dones, next_value)
        
        # Normalize advantages
        if self.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update value normalizer
        if self.normalize_value:
            self.value_normalizer.update(returns)
            normalized_returns = self.value_normalizer.normalize(returns)
        else:
            normalized_returns = returns
        
        # Convert to tensors
        observations = torch.FloatTensor(observations).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(normalized_returns).to(self.device)
        
        # Training statistics
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        n_updates = 0
        
        # Multiple epochs of training
        T = len(observations)
        indices = np.arange(T)
        
        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, T, self.batch_size):
                end = min(start + self.batch_size, T)
                batch_indices = indices[start:end]
                
                # Get batch
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Update each agent's actor
                actor_losses = []
                entropies = []
                approx_kls = []
                
                for agent_idx in range(self.num_agents):
                    agent_obs = batch_obs[:, agent_idx]
                    agent_actions = batch_actions[:, agent_idx]
                    agent_old_log_probs = batch_old_log_probs[:, agent_idx]
                    
                    # Get new log probs and entropy
                    logits = self.actors[agent_idx](agent_obs)
                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(agent_actions)
                    entropy = dist.entropy().mean()
                    
                    # Compute policy loss with PPO clipping
                    ratio = torch.exp(new_log_probs - agent_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # Add entropy bonus
                    actor_loss = actor_loss - self.ent_coef * entropy
                    
                    # Update actor
                    self.actor_optimizers[agent_idx].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), self.max_grad_norm)
                    self.actor_optimizers[agent_idx].step()
                    
                    actor_losses.append(actor_loss.item())
                    entropies.append(entropy.item())
                    
                    # Compute approximate KL divergence
                    with torch.no_grad():
                        approx_kl = (agent_old_log_probs - new_log_probs).mean()
                        approx_kls.append(approx_kl.item())
                
                # Update centralized critic
                global_states = batch_obs.reshape(len(batch_obs), -1)
                pred_values = self.critic(global_states).squeeze(-1)
                critic_loss = nn.functional.mse_loss(pred_values, batch_returns)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                # Accumulate statistics
                total_actor_loss += np.mean(actor_losses)
                total_critic_loss += critic_loss.item()
                total_entropy += np.mean(entropies)
                total_approx_kl += np.mean(approx_kls)
                n_updates += 1
        
        # Update learning rates
        for scheduler in self.actor_schedulers:
            scheduler.step()
        self.critic_scheduler.step()
        
        self.update_count += 1
        
        # Return training statistics
        return {
            'actor_loss': total_actor_loss / n_updates,
            'critic_loss': total_critic_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'approx_kl': total_approx_kl / n_updates,
            'actor_lr': self.actor_optimizers[0].param_groups[0]['lr'],
            'critic_lr': self.critic_optimizer.param_groups[0]['lr']
        }
    
    def save(self, path: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critic': self.critic.state_dict(),
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'value_normalizer': self.value_normalizer.__dict__ if self.normalize_value else None,
            'update_count': self.update_count
        }, path)
    
    def load(self, path: str):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint['actors'][i])
        self.critic.load_state_dict(checkpoint['critic'])
        for i, opt in enumerate(self.actor_optimizers):
            opt.load_state_dict(checkpoint['actor_optimizers'][i])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        if self.normalize_value and checkpoint['value_normalizer']:
            self.value_normalizer.__dict__.update(checkpoint['value_normalizer'])
        self.update_count = checkpoint['update_count']
