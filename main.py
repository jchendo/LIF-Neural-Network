import DigitDraw
import LIFNeuron
import numpy as np
import pickle
import sys

laptop = True
if laptop:
    FILEPATH = "C:/Users/jacob/OneDrive/Desktop/Code/Projects/DigitRec Neural Network/"
else:
    FILEPATH = "C:/Users/jacob/Desktop/Code/Python/DigitRec/"

NUM_INPUT_LAYER = 784
NUM_HIDDEN_LAYER = 200
NUM_OUTPUT_LAYER = 10
NUM_TIMESTEPS = 50

digit_drawing = DigitDraw.DigitDraw()
running = True

class LIFNeuralNetwork:

    def __init__(self):
        self.input_layer_neurons = None
        self.hidden_layer_neurons = None
        self.output_layer_neurons = None

        self.input_hidden_connections = {} ## LIFNeuron: [LIFNeuron, LIFNeuron, LIFNeuron, etc] -> the key receives input from the list of neurons
        self.input_hidden_weights = {} ## {LIFNeuron: [0.1,0.2,0.3,0.4]} etc

        self.hidden_output_connections = {}
        self.hidden_output_weights = {} 

    def updateNeurons(self, timestep):
        pixels = digit_drawing.pixels
        ## update input neurons
        for num in range(NUM_INPUT_LAYER): ## probably numpy vectorize this
            neuron = self.input_layer_neurons[num]

            pixel_val_x = num % 28
            pixel_val_y = num // 28
            pixel = pixels[pixel_val_x][pixel_val_y]

            ## could potentially do this elsewhere since it only needs to be done once
            if pixel: ## colored black
                neuron.input = 1

            neuron.update(timestep)

        ## update hidden neurons
        for neuron in self.hidden_layer_neurons:
            neuron.input = 0
            self.sumInputs(neuron)
            neuron.update(timestep)

        ## update output neurons
        for neuron in self.output_layer_neurons:
            neuron.input = 0
            self.sumInputs(neuron)
            neuron.update(timestep)

    def sumInputs(self,neuron):

        if neuron.type == 1:
            neurons = self.input_layer_neurons
            connections = self.input_hidden_connections
            weights = self.input_hidden_weights
        else:
            neurons = self.hidden_layer_neurons
            connections = self.hidden_output_connections
            weights = self.hidden_output_weights

        for input_num in range(len(connections[neuron.id])):
            input_neuron_id = connections[neuron.id][input_num]
            input_neuron = neurons[input_neuron_id]
            synapse_weight = weights[neuron.id][input_num]   

            # if neuron.type == 2:
            #     print(input_neuron.output)

            neuron.input += input_neuron.output * synapse_weight
        
    def connectNeurons(self):

        self.input_layer_neurons = np.array([LIFNeuron.LIF_Neuron(type=0, id=id, delay=np.random.randint(1,4)) for id in range(NUM_INPUT_LAYER)])
        self.hidden_layer_neurons = np.array([LIFNeuron.LIF_Neuron(type=1, id=id, delay=0) for id in range(NUM_HIDDEN_LAYER)])
        self.output_layer_neurons = np.array([LIFNeuron.LIF_Neuron(type=2, id=id, delay=0) for id in range(NUM_OUTPUT_LAYER)])

        num_hidden_to_output = NUM_HIDDEN_LAYER ## since there's only 10 output neurons, maybe change this

        ## Connect hidden layer neurons to input neurons
        for neuron in self.hidden_layer_neurons:
            num_input_to_hidden = int(np.random.uniform(0.1, 0.3) * NUM_HIDDEN_LAYER)
            connections = []
            weights = np.random.uniform(0.01, 0.05, size=num_input_to_hidden)
            for _ in range(num_input_to_hidden): 
                connections.append(np.random.choice(self.input_layer_neurons).id)

            self.input_hidden_connections[neuron.id] = connections
            self.input_hidden_weights[neuron.id] = weights

        ## Output layer to hidden
        for neuron in self.output_layer_neurons:
            ## We're just connecting every hidden layer neuron to every output neuron
            connections = [_ for _ in range(0,200)]
            weights = np.random.uniform(0.01, 0.05, size=num_hidden_to_output)

            self.hidden_output_connections[neuron.id] = connections
            self.hidden_output_weights[neuron.id] = weights

        self.saveConnectivity()
        return self
    
    @classmethod
    def loadConnectivity(cls):
        try:
            if input("New network? ") == 'Y':
                return None ## goes to the except block
            with open(f"{FILEPATH}/data/network.pkl", "rb") as f:
                return pickle.load(f)
        except:
            print("Import failed. Generate new network? Y/N")
            if input("") == 'Y':
                return None
            else:
                sys.exit()

    def saveConnectivity(self):
        with open(f"{FILEPATH}/data/network.pkl", "wb") as f:
            pickle.dump(self, f)

    def run(self):
        digit_drawing.begin_drawing()

        if digit_drawing.done:
            for timestep in range(NUM_TIMESTEPS):
                self.updateNeurons(timestep)
        else:
            print("No number drawn. Exiting...")
            sys.exit()

if __name__ == "__main__":
    neural_net = LIFNeuralNetwork.loadConnectivity()
    #print(neural_net.input_hidden_connections)
    if neural_net == None:
        neural_net = LIFNeuralNetwork()
        neural_net.connectNeurons()
    neural_net.run()
