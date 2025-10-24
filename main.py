import DigitDraw
import LIFNeuron
import NetworkDisplay
import MNISTLoader # type: ignore
from os.path import join
import numpy as np
import pickle
import sys

laptop = True
if laptop:
    FILEPATH = "C:/Users/jacob/OneDrive/Desktop/Code/Projects/DigitRec Neural Network/"
else:
    FILEPATH = "C:/Users/jacob/Desktop/Code/Python/DigitRec/"

data_root          = join(FILEPATH, "data/MNIST Dataset")
training_images_fp = join(data_root, "train-images.idx3-ubyte")
training_labels_fp = join(data_root, "train-labels.idx1-ubyte")
test_images_fp     = join(data_root, "t10k-images.idx3-ubyte")
test_labels_fp     = join(data_root, "t10k-labels.idx1-ubyte")

NUM_INPUT_LAYER = 784
NUM_HIDDEN_LAYER = 200
NUM_OUTPUT_LAYER = 10
NUM_TIMESTEPS = 500

digit_drawing = DigitDraw.DigitDraw()

class LIFNeuralNetwork:

    def __init__(self):
        self.input_layer_neurons = None
        self.hidden_layer_neurons = None
        self.output_layer_neurons = None
        self.ALL_NEURONS = None

        self.input_hidden_connections = {} ## LIFNeuron: [LIFNeuron, LIFNeuron, LIFNeuron, etc] -> the key receives input from the list of neurons
        self.input_hidden_weights = {} ## {LIFNeuron: [0.1,0.2,0.3,0.4]} etc

        self.hidden_output_connections = {}
        self.hidden_output_weights = {} 

    def updateNeurons(self, timestep):
        pixels = digit_drawing.pixels
        ## update input neurons
        for num in range(NUM_INPUT_LAYER): ## probably numpy vectorize this
            neuron = self.input_layer_neurons[num]

            pixel_val_x = num % digit_drawing.NUM_PIXELS
            pixel_val_y = num // digit_drawing.NUM_PIXELS
            pixel = pixels[pixel_val_x][pixel_val_y]

            ## could potentially do this elsewhere since it only needs to be done once
            if pixel: ## colored black
                neuron.input = pixel / 255.0 

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
        
        match neuron.type:
            case 1: ## hidden layer
                    neurons = self.input_layer_neurons
                    connections = self.input_hidden_connections
                    weights = self.input_hidden_weights
            case 2: ## output layer
                    neurons = self.hidden_layer_neurons
                    connections = self.hidden_output_connections
                    weights = self.hidden_output_weights
    
        for input_num in range(len(connections[neuron.id])):
            input_neuron_id = connections[neuron.id][input_num]
            input_neuron = neurons[input_neuron_id]
            synapse_weight = weights[neuron.id][input_num]   

            neuron.input += input_neuron.output * synapse_weight
        
    def predictNumber(self):
        num_total_spikes = 0
        confidence = 0.0
        guess = None
        ## sum total number of spikes
        for output_neuron in self.output_layer_neurons: 
            num_local_spikes = np.sum(output_neuron.spikes)
            num_total_spikes += num_local_spikes
        ## compare confidence values
        for num in range(len(self.output_layer_neurons)):
            output_neuron = self.output_layer_neurons[num]
            num_local_spikes = np.sum(output_neuron.spikes)
            local_confidence = float(num_local_spikes)/num_total_spikes

            print(num_local_spikes)

            if local_confidence > confidence:
                guess = num
                confidence = local_confidence

        return guess
    
    def clearNeuronHistories(self):
        for neuron in self.ALL_NEURONS:
            neuron.spikes = np.zeros(NUM_TIMESTEPS)
            neuron.voltages = np.zeros(NUM_TIMESTEPS)
    
    def compileNetworkData(self):
        data = [self.input_layer_neurons, self.hidden_layer_neurons, self.output_layer_neurons, self.input_hidden_connections]
        return data

    def connectNeurons(self):

        self.input_layer_neurons = np.array([LIFNeuron.LIF_Neuron(type=0, id=id, delay=np.random.randint(1,4)) for id in range(NUM_INPUT_LAYER)])
        self.hidden_layer_neurons = np.array([LIFNeuron.LIF_Neuron(type=1, id=id, delay=0) for id in range(NUM_HIDDEN_LAYER)])
        self.output_layer_neurons = np.array([LIFNeuron.LIF_Neuron(type=2, id=id, delay=0) for id in range(NUM_OUTPUT_LAYER)])
        self.ALL_NEURONS = np.concatenate([self.input_layer_neurons, self.hidden_layer_neurons, self.output_layer_neurons])

        num_hidden_to_output = NUM_HIDDEN_LAYER ## since there's only 10 output neurons, maybe change this

        ## Connect hidden layer neurons to input neurons
        for neuron in self.hidden_layer_neurons:
            num_input_to_hidden = int(np.random.uniform(0.1, 0.3) * NUM_HIDDEN_LAYER)
            connections = []
            weights = np.random.uniform(0.02, 0.05, size=num_input_to_hidden)
            for _ in range(num_input_to_hidden): 
                connections.append(np.random.choice(self.input_layer_neurons).id)

            self.input_hidden_connections[neuron.id] = connections
            self.input_hidden_weights[neuron.id] = weights

        ## Output layer to hidden
        for neuron in self.output_layer_neurons:
            ## We're just connecting every hidden layer neuron to every output neuron
            connections = [_ for _ in range(NUM_HIDDEN_LAYER)]
            weights = np.random.uniform(0.1, 0.35, size=num_hidden_to_output)

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
        self.clearNeuronHistories()
        digit_drawing.begin_drawing()

        if digit_drawing.done:
            for timestep in range(NUM_TIMESTEPS):
                self.updateNeurons(timestep)

            # network_display = NetworkDisplay.NetworkDisplay() ## have to initialize this here otherwise it gets weird w/ DigitDraw
            # network_data = self.compileNetworkData()
            # network_display.run(network_data, NUM_TIMESTEPS)

            guess = self.predictNumber()
            print(f"Guess: {guess}")

        else:
            print("No number drawn. Exiting...")
            sys.exit()

if __name__ == "__main__":
    ## MNIST data
    data_loader = MNISTLoader.MNISTDataloader(training_images_fp, training_labels_fp, test_images_fp, test_labels_fp)
    (training_images, training_labels), (test_images, test_labels) = data_loader.load_data()

    neural_net = LIFNeuralNetwork.loadConnectivity()
    if neural_net == None:
        neural_net = LIFNeuralNetwork()
        neural_net.connectNeurons()
    neural_net.run()
