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