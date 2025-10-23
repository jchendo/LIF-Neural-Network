import numpy as np

SPIKE_THR = -50
TIMESTEPS = 50
TC = 10 ## time constant, subject to change

class LIF_Neuron:
    type          = 0 ## 0,1,2 -> input, hidden, output
    output        = 0 ## 0/1
    input         = 0.0
    delay         = 0 ## total amount of time to delay firing (prevents a wall of spikes)
    curr_delay    = 0 ## current amount of delay that has passed
    spike_str     = 0
    id            = 0
    Vm            = -65
    spikes        = np.zeros(TIMESTEPS, dtype=np.int8)

    def __init__(self, type, id, delay):
        self.type = type
        self.id = id
        self.delay = delay
        match self.type:
            case 0: self.spike_str = 20
            case 1: self.spike_str = 200
            case 2: self.spike_str = 20

    def update(self, timestep):
        self.Vm += (1.0/TC) * (-(self.Vm + 65) + (self.input * self.spike_str))

        if self.type == 0:
            #print(self.spike_str)
            print(f"Vm: {self.Vm}")

        if self.Vm >= SPIKE_THR and self.curr_delay >= self.delay:
            self.emit_spike()
            self.Vm = -65
            self.spikes[timestep] = 1
            self.curr_delay = 0

        elif self.Vm >= SPIKE_THR: ## be careful with this delay, definitely a possbility this logic could cook the input/output encoding
            self.curr_delay += 1
        else:
            self.output = 0

    def emit_spike(self):
        self.output = 1

