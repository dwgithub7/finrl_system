# ppo_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.policy = nn.Sequential(
            nn.Linear(128, action_dim), nn.Softmax(dim=-1)
        )
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        shared = self.shared(x)
        return self.policy(shared), self.value(shared)

class PPOAgent:
    def __init__(self, state_dim, action_dim, clip_eps=0.2, lr=3e-4):
        self.model = PPOActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.gamma = 0.99
        self.lmbda = 0.95
        self.entropy_coef = 0.01
        self.value_coef = 0.5

    def get_action(self, state):
        state_tensor = torch.tensor(state).unsqueeze(0).float()
        probs, _ = self.model(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def compute_gae(self, rewards, values, next_value, dones):
        gae = 0
        returns = []
        values = values + [next_value]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.lmbda * (1 - dones[step]) * gae
            returns.insert(0, gae + values[step])
        return returns

    def update(self, trajectories):
        states = torch.tensor(np.vstack([t[0] for t in trajectories]), dtype=torch.float32)
        actions = torch.tensor([t[1] for t in trajectories])
        old_log_probs = torch.stack([t[2] for t in trajectories])
        rewards = [t[3] for t in trajectories]
        dones = [t[4] for t in trajectories]
        values = [t[5] for t in trajectories]
        next_value = trajectories[-1][6]
        returns = self.compute_gae(rewards, values, next_value, dones)
        returns = torch.tensor(returns, dtype=torch.float32)

        for _ in range(4):
            probs, values_est = self.model(states)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_log_probs.detach())
            adv = returns - values_est.squeeze()
            clip_adv = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv

            actor_loss = -torch.min(ratio * adv, clip_adv).mean()
            critic_loss = F.mse_loss(values_est.squeeze(), returns)
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
