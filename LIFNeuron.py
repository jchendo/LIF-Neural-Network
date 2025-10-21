import numpy as np

SPIKE_THR = -50
TIMESTEPS = 50
TC = 10 ## time constant, subject to change

class LIF_Neuron:
    type          = 0 ## 0,1,2 -> input, hidden, output
    output        = 0 ## 0/1
    input         = 0
    spike_str     = 0
    Vm            = -65

    def __init__(self, type):
        self.type = type
        match self.type:
            case 0: self.spike_str = 20
            case 1: self.spike_str = 5
            case 2: self.spike_str = 5

    def update(self):
        self.output = 0
        self.Vm += (1.0/TC) * (-(self.Vm + 65) + (self.input * self.spike_str))

        if self.Vm >= SPIKE_THR:
            self.emit_spike()
            self.Vm = -65

    def emit_spike(self):
        print("Spiked")
        self.output = 1

