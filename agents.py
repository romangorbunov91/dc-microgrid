class PowerSource:
    def __init__(self, Rdroop, Uinit, dt, tau=None):
        self.Rdroop = Rdroop
        self.u_int_prev = Uinit
        if tau is None:
            self.alpha = None
        else:
            self.alpha = dt/tau

    def step(self, i, Uref):
        if self.alpha is None:
            u_int = Uref
        else:
            u_int = self.alpha * Uref + (1 - self.alpha) * self.u_int_prev
            self.u_int_prev = u_int
        return u_int - i * self.Rdroop

class PowerLoad:
    def __init__(self, Unom, Pnom, Rline, type, tau=None):
        
        self.Rline = Rline
        self.tau = tau
        self.u_int_prev = 0.0
        
        match type:
            case 'current':
                self.I = Pnom/Unom
            case 'res':
                self.R = Unom**2 / Pnom

    def step(self, u, dt):
        if self.tau is not None:
            self.u_prev += (self.Uref - self.us_prev) * dt / self.tau
            u_int = self.u_int_prev
        else:
            u_s = self.Uref
        return u - i * self.Rdroop