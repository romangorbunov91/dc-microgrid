class PowerSource:
    def __init__(self, Rdroop, Uinit=0.0, beta=1.0):
        self.Rdroop = Rdroop
        self.u = Uinit
        self.beta = beta

    def step(self, u, Uref):
        self.u = self.beta * Uref + (1 - self.beta) * self.u
        return (self.u - u) / self.Rdroop

class PowerLoad:
    def __init__(self, Unom, Pnom, Iinit=0.0, beta=1.0):      
        self.R = Unom**2 / Pnom
        self.beta = beta
        self.i = Iinit

    def step(self, u):
        self.i = self.beta * u / self.R + (1 - self.beta) * self.i
        return self.i
    
class PowerBus:
    def __init__(self, Uinit=0.0, beta=1.0):      
        self.beta = beta
        self.u = Uinit

    def step(self, i):
        self.u += self.beta * i
        return self.u

class PowerGrid:
    def __init__(self,
                 Ubus_init,
                 beta_bus,
                 N_SOURCE,
                 Rdroop_set,
                 Usource_set,
                 beta_source_set,
                 N_LOAD,
                 Uload_set,
                 Pload_set,
                 beta_load_set):      
        
        self.bus = PowerBus(
            Uinit=Ubus_init,
            beta=beta_bus
            )

        self.sources = [
            PowerSource(
                Rdroop=Rdroop_set[idx],
                Uinit=Usource_set[idx],
                beta=beta_source_set[idx]
                ) for idx in range(N_SOURCE)
        ]

        self.loads = [
            PowerLoad(
                Unom=Uload_set[idx],
                Pnom=Pload_set[idx],
                beta=beta_load_set[idx]
                ) for idx in range(N_LOAD)
        ]
        self.Ubus_prev = Ubus_init
    def step(self, N_STEPS, Uref_set):
        source_currents = np.zeros((N_STEPS, len(self.sources)))
        load_currents = np.zeros((N_STEPS, len(self.loads)))
        bus_voltage = np.zeros(N_STEPS)

        for n in range(N_STEPS):
            if n == 0:
                source_currents[n] = [source.step(u=self.Ubus_prev, Uref=Uref_set[idx])
                                for idx, source in enumerate(self.sources)]
                load_currents[n] = [load.step(u=self.Ubus_prev)
                                for load in self.loads]
            else:
                source_currents[n] = [source.step(u=bus_voltage[n-1], Uref=Uref_set[idx])
                                for idx, source in enumerate(self.sources)]
                load_currents[n] = [load.step(u=bus_voltage[n-1])
                                for load in self.loads]
            
            bus_voltage[n] = self.bus.step(source_currents[n].sum() - load_currents[n].sum())
        
        # Save last bus voltage value as an initial for next '.step'.
        self.Ubus_prev = bus_voltage[-1]
        return bus_voltage, source_currents, load_currents


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
            nn.Linear(hidden_dim, action_dim * 2)
        )
        self.action_dim = action_dim

    def forward(self, obs):
        out = self.net(obs)
        mean, log_std = out[:, :self.action_dim], out[:, self.action_dim:]
        log_std = torch.clamp(log_std, -20, 2)
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
        return self.net(state).squeeze(-1)