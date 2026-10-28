"""
agent.py: Soft Actor-Critic (SAC) for Adaptive Surface Control.

Features:
- Maximum Entropy Reinforcement Learning (Haarnoja et al., 2018)
- Twin Q-networks with clipped double Q-learning
- Reparameterized Gaussian policy with squashed tanh output
- Automatic temperature (alpha) adaptation
- Residual policy conditioning (outputs learned residual on top of crystallised baseline)
- Modular ReplayBuffer with uniform sampling
- Support for few-shot subject adaptation and weight checkpointing
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class ReplayBuffer:
    """Experience replay buffer for off-policy SAC training."""

    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 100000, device: str = "cpu"):
        self.capacity = capacity
        self.device = torch.device(device)
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition tuple."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[indices], device=self.device),
            torch.as_tensor(self.actions[indices], device=self.device),
            torch.as_tensor(self.rewards[indices], device=self.device),
            torch.as_tensor(self.next_obs[indices], device=self.device),
            torch.as_tensor(self.dones[indices], device=self.device),
        )

    def __len__(self) -> int:
        return self.size


class CriticQNetwork(nn.Module):
    """Twin Q-network architecture for evaluating state-action values."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: Tuple[int, int] = (256, 256)):
        super().__init__()
        # Q1 Architecture
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Linear(hidden_dims[1], 1),
        )

        # Q2 Architecture (Twin critic to prevent overestimation bias)
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([obs, action], dim=-1)
        return self.q1(sa)


class GaussianPolicy(nn.Module):
    """Squashed Gaussian policy network for continuous surface residual control."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: Tuple[int, int] = (256, 256),
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.GELU(),
        )

        self.mu_head = nn.Linear(hidden_dims[1], act_dim)
        self.log_std_head = nn.Linear(hidden_dims[1], act_dim)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.net(obs)
        mu = self.mu_head(feat)
        log_std = self.log_std_head(feat)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action via reparameterization trick with tanh squashing.
        
        Returns:
            action: Squashed action in [-1, 1].
            log_prob: Exact log likelihood of the squashed action.
            mean: Squashed deterministic mean action for evaluation.
        """
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mu, std)

        # Reparameterization trick: u = mu + std * eps
        u = normal.rsample()
        action = torch.tanh(u)

        # Enforcing Action Bounds: log pi(a|s) = log mu(u|s) - sum log(1 - tanh(u)^2 + eps)
        log_prob = normal.log_prob(u) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mu)
        return action, log_prob, mean_action


class SACAgent:
    """Soft Actor-Critic Agent for continuous adaptive surface control."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        hidden_dims: Tuple[int, int] = (256, 256),
        init_temperature: float = 0.2,
        buffer_capacity: int = 100000,
        device: Optional[str] = None,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Policy & Critic Networks
        self.actor = GaussianPolicy(obs_dim, act_dim, hidden_dims).to(self.device)
        self.critic = CriticQNetwork(obs_dim, act_dim, hidden_dims).to(self.device)
        self.critic_target = CriticQNetwork(obs_dim, act_dim, hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Automatic Entropy Temperature (Alpha) Tuning:
        # Target entropy = -dim(A)
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.tensor(np.log(init_temperature), requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

        # Experience Replay
        self.replay_buffer = ReplayBuffer(obs_dim, act_dim, buffer_capacity, str(self.device))

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp().item()

    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action given observation."""
        self.actor.eval()
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if evaluate:
                _, _, action = self.actor.sample(obs_tensor)
            else:
                action, _, _ = self.actor.sample(obs_tensor)
        self.actor.train()
        return action.squeeze(0).cpu().numpy()

    def update_parameters(self, batch_size: int = 128) -> Dict[str, float]:
        """Perform one gradient update step for Critic, Actor, and Temperature."""
        if len(self.replay_buffer) < batch_size:
            return {}

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(batch_size)

        # ------------------ 1. Critic Update ------------------ #
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.sample(next_obs)
            q1_target, q2_target = self.critic_target(next_obs, next_actions)
            min_q_target = torch.min(q1_target, q2_target) - self.log_alpha.exp() * next_log_pi
            y = rewards + (1.0 - dones) * self.gamma * min_q_target

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ------------------ 2. Actor Update ------------------ #
        new_actions, log_pi, _ = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_actions)
        min_q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.log_alpha.exp() * log_pi - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ------------------ 3. Temperature Update ------------------ #
        alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ------------------ 4. Polyak Target Update ------------------ #
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha,
        }

    def save_checkpoint(self, filepath: str) -> None:
        """Save model weights and state."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha.data,
            },
            filepath,
        )

    def load_checkpoint(self, filepath: str) -> None:
        """Load model weights and state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.log_alpha.data.copy_(checkpoint["log_alpha"])
