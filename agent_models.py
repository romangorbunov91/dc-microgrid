import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            #nn.Linear(hidden_dim, action_dim * 2)
            nn.Linear(hidden_dim, action_dim)
            
        )
        self.action_dim = action_dim

    def forward(self, obs):
        mean = self.net(obs.float())
        #mean, log_std = out[:, :self.action_dim], out[:, self.action_dim:]
        #log_std = torch.clamp(log_std, -20, 2)
        log_std = torch.tensor(1.1, dtype=torch.float32)
        return mean, log_std

    def get_action(self, obs, deterministic=False):
        mean, log_std = self.forward(obs)
        if deterministic:
            return torch.tanh(mean), None
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        action = torch.tanh(action)
        # Correction for tanh squashing
        log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(-1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, global_state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)#.squeeze(-1)