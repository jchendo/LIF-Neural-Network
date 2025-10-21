import numpy as np

SPIKE_STR = 1 ## how much each spikes add to membrane voltage
SPIKE_THR = -50
TIMESTEPS = 50
TC = 10 ## time constant, subject to change

class LIF_Neuron:
    type          = 0 ## 0,1,2 -> input, hidden, output
    output        = 0 ## 0/1
    input         = 0
    Vm            = -65

    def __init__(self, type):
        self.type = type

    def update(self):
        self.Vm += (1.0/TC) * (-(self.Vm + 65) + (self.input * SPIKE_STR))
        if self.Vm >= SPIKE_THR:
            self.emit_spike(self)
            self.Vm = -65

    def emit_spike(self):
        print("Spiked")
        self.output = 1

