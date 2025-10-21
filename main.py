import DigitDraw
import LIFNeuron
import numpy as np
import pickle
import sys

laptop = False
if laptop:
    FILEPATH = "C:/Users/jacob/OneDrive/Desktop/Code/Projects/DigitRec Neural Network/"
else:
    FILEPATH = "C:/Users/jacob/Desktop/Code/Python/DigitRec/"

NUM_INPUT_LAYER = 784
NUM_HIDDEN_LAYER = 200
NUM_OUTPUT_LAYER = 10

digit_drawing = DigitDraw.DigitDraw()
running = True

class LIFNeuralNetwork:

    input_layer_neurons = None
    hidden_layer_neurons = None
    output_layer_neurons = None

    input_hidden_connections = {} ## LIFNeuron: [LIFNeuron, LIFNeuron, LIFNeuron, etc] -> the key receives input from the list of neurons
    input_hidden_weights = {} ## {LIFNeuron: [0.1,0.2,0.3,0.4]} etc

    hidden_output_connections = {}
    hidden_output_weights = {} 

    def updateNeurons(self):
        pixels = digit_drawing.pixels
        for num in range(NUM_INPUT_LAYER): ## probably numpy vectorize this
            neuron = self.input_layer_neurons[num]
            neuron.input = 0 ## reset

            pixel_val_x = num % 28
            pixel_val_y = num // 28
            pixel = pixels[pixel_val_x][pixel_val_y]

            if pixel: ## colored black
                neuron.input = 1

            neuron.update()

        for neuron in self.hidden_layer_neurons:
            neuron.input = 0
            self.sumInputs(neuron, 1)

    def sumInputs(self,neuron, type):

        if type == 1:
            connections = self.input_hidden_connections
            weights = self.input_hidden_weights
        else:
            connections = self.hidden_output_connections
            weights = self.hidden_output_weights

        for input_num in range(len(connections[neuron])):
            input_neuron = connections[neuron][input_num]
            synapse_weight = weights[neuron][input_num]

            neuron.input += input_neuron.output * synapse_weight
        
    def connectNeurons(self):

        self.input_layer_neurons = np.array([LIFNeuron.LIF_Neuron(type=0) for _ in range(NUM_INPUT_LAYER)])
        self.hidden_layer_neurons = np.array([LIFNeuron.LIF_Neuron(type=1) for _ in range(NUM_HIDDEN_LAYER)])
        self.output_layer_neurons = np.array([LIFNeuron.LIF_Neuron(type=2) for _ in range(NUM_OUTPUT_LAYER)])

        num_input_to_hidden = int(np.random.uniform(0.1, 0.3) * NUM_HIDDEN_LAYER)
        num_hidden_to_output = NUM_OUTPUT_LAYER ## since there's only 10 output neurons, maybe change this

        for neuron in self.hidden_layer_neurons:
            connections = []
            weights = np.random.uniform(0.01, 0.05, size=num_input_to_hidden)
            for _ in range(num_input_to_hidden): 
                connections.append(np.random.choice(self.input_layer_neurons))

            self.input_hidden_connections[neuron] = connections
            self.input_hidden_weights[neuron] = weights

        for neuron in self.output_layer_neurons:
            connections = []
            weights = np.random.uniform(0.01, 0.05, size=num_hidden_to_output)
            for _ in range(num_hidden_to_output): 
                connections.append(np.random.choice(self.hidden_layer_neurons))

            self.hidden_output_connections[neuron] = connections
            self.hidden_output_weights[neuron] = weights

        self.saveConnectivity()
        return self

    def saveConnectivity(self):
        with open(f"{FILEPATH}/data/network.pkl", "wb") as f:
            pickle.dump(self, f)

    def run(self):
        digit_drawing.begin_drawing()

        if digit_drawing.done:
            while running:
                self.updateNeurons()
        else:
            print("No number drawn. Exiting...")
            sys.exit()

if __name__ == "__main__":
    neural_net = LIFNeuralNetwork()
    try:
        with open(f"{FILEPATH}/data/network.pkl", "rb") as f:
            data = pickle.load(f)
            neural_net = data
    except:
        print("Import failed. Generate new network? Y/N")
        if input("") == 'Y':
            neural_net.connectNeurons()
        else:
            sys.exit()

    neural_net.run()
