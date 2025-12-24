import numpy as np
    
class PowerBus:
    def __init__(self, Uinit=0.0, beta=1.0):      
        self.beta = beta
        self.u = Uinit

    def step(self, i):
        self.u += self.beta * i
        return self.u

class PowerSource:
    def __init__(self, Rdroop=1e-3, Uinit=0.0, beta=1.0):
        self.Rdroop = Rdroop
        self.u = Uinit
        self.beta = beta

    def step(self, u, Uref):
        self.u = self.beta * Uref + (1 - self.beta) * self.u
        return (self.u - u) / self.Rdroop, self.u

class PowerLoad:
    def __init__(self, Unom, Iinit=0.0, beta=1.0):      
        self.Unom = Unom
        self.i = Iinit
        self.beta = beta

    def step(self, u, P):
        R = self.Unom**2 / P
        self.i = self.beta * u / R + (1 - self.beta) * self.i
        return self.i

class PowerGrid:
    def __init__(self,
                 N_SOURCE,
                 N_LOAD,
                 Rdroop_set,
                 Ubus_init,
                 Usource_init_set,
                 Uload_nom_set,
                 Iload_init_set,
                 beta_bus,
                 beta_source_set,
                 beta_load_set):      
        
        self.bus = PowerBus(
            Uinit = Ubus_init,
            beta = beta_bus
            )

        self.sources = [
            PowerSource(
                Rdroop = Rdroop_set[idx],
                Uinit = Usource_init_set[idx],
                beta = beta_source_set[idx]
                ) for idx in range(N_SOURCE)
        ]

        self.loads = [
            PowerLoad(
                Unom = Uload_nom_set[idx],
                Iinit = Iload_init_set[idx],
                beta = beta_load_set[idx]
                ) for idx in range(N_LOAD)
        ]
        
        # First iteration use initial value.
        self.Ubus_prev = Ubus_init

    def step(self, N_STEPS, Uref_set, Pload_set):
        source_currents = np.zeros((N_STEPS, len(self.sources)))
        source_voltages = np.zeros((N_STEPS, len(self.sources)))
        load_currents = np.zeros((N_STEPS, len(self.loads)))
        bus_voltage = np.zeros(N_STEPS)

        for n in range(N_STEPS):
            if n == 0:
                IU_source = [
                    source.step(
                        u = self.Ubus_prev,
                        Uref = Uref_set[idx]
                        ) for idx, source in enumerate(self.sources)
                    ]
                load_currents[n] = [
                    load.step(
                        u = self.Ubus_prev,
                        P = Pload_set[idx]
                        ) for idx, load in enumerate(self.loads)]
            else:
                IU_source = [
                    source.step(
                        u = bus_voltage[n-1],
                        Uref = Uref_set[idx]
                        ) for idx, source in enumerate(self.sources)
                    ]
                load_currents[n] = [
                    load.step(
                        u = bus_voltage[n-1],
                        P = Pload_set[idx]
                        ) for idx, load in enumerate(self.loads)]
            
            for idx, (I, U) in enumerate(IU_source):
                source_currents[n,idx] = I
                source_voltages[n,idx] = U  
            bus_voltage[n] = self.bus.step(source_currents[n].sum() - load_currents[n].sum())
        
        # Save last bus voltage value as an initial for next '.step'.
        self.Ubus_prev = bus_voltage[-1]
        
        # Compose outputs.
        transients = [
            bus_voltage,
            source_currents,
            source_voltages,
            load_currents
            ]
        
        steady_states = [
            bus_voltage[-1],
            source_currents[-1],
            source_voltages[-1],
            load_currents[-1]
            ]
        
        # Observed states by each agent.
        # [I_self, I_left, I_right, U_self, U_left, U_right].
        IU_sets = [source_currents[-1], source_voltages[-1]]
        N_NEIGH = 2
        
        agent_obs_states = np.zeros((len(self.sources), len(IU_sets) * (N_NEIGH+1)))
        
        for idx in range(len(self.sources)):
            state_list = []
            for var in IU_sets:
                # [self, left, right]
                for m in [0, -1, 1]:
                    state_list.append(var[(idx+m)%len(self.sources)])
            agent_obs_states[idx,:] = state_list
        
        # Global state of the grid. 
        global_state = np.zeros(len(IU_sets) * len(self.sources))
        for m, var in enumerate(IU_sets):
            for k, value in enumerate(var):
                global_state[m*len(var)+k] = value
        
        return transients, steady_states, agent_obs_states, global_state