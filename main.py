import DigitDraw
import LIFNeuron
import NetworkDisplay
import MNISTLoader # type: ignore
import numpy as np
import pickle
import sys
from os.path import join
from time import time

laptop = False
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
NUM_TIMESTEPS = 50
NUM_EPOCHS = 10
LEARNING_RATE = 0.01

digit_drawing = DigitDraw.DigitDraw()

class LIFNeuralNetwork:

    def __init__(self):
        self.time_accumulated   = 0
        self.input_layer_neurons = None
        self.hidden_layer_neurons = None
        self.output_layer_neurons = None
        self.ALL_NEURONS = None

        self.input_layer_output = np.zeros(NUM_INPUT_LAYER)
        self.hidden_layer_output = np.zeros(NUM_HIDDEN_LAYER) ## store all the outputs as one list to limit indexing times
        
        self.input_layer_eligibility = np.zeros(NUM_INPUT_LAYER)
        self.hidden_layer_eligibility = np.zeros(NUM_HIDDEN_LAYER)

        self.input_hidden_connections = [] ##  -> the key receives input from the list of neurons
        self.input_hidden_weights = [] ## {LIFNeuron: [0.1,0.2,0.3,0.4]} etc

        self.hidden_output_connections = []
        self.hidden_output_weights = []

    def updateNeurons(self, timestep):
        ## update input neurons
        for neuron in self.input_layer_neurons: ## probably numpy vectorize this
            neuron.update(timestep)
            ## Use these to vectorize input & weight updating
            self.input_layer_output[neuron.id] = neuron.output
            self.input_layer_eligibility[neuron.id] = neuron.eligibility

        ## update hidden neurons
        for neuron in self.hidden_layer_neurons:
            neuron.input = 0
            self.sumInputs(neuron)
            neuron.update(timestep)

            self.hidden_layer_output[neuron.id] = neuron.output
            self.hidden_layer_eligibility[neuron.id] = neuron.eligibility

        ## update output neurons
        for neuron in self.output_layer_neurons:
            neuron.input = 0
            self.sumInputs(neuron)
            neuron.update(timestep)

    def sumInputs(self,neuron):
        start_time = time()
        match neuron.type:
            case 1: ## hidden layer
                    outputs = self.input_layer_output
                    connections = self.input_hidden_connections
                    weights = self.input_hidden_weights
            case 2: ## output layer
                    outputs = self.hidden_layer_output
                    connections = self.hidden_output_connections
                    weights = self.hidden_output_weights
    
        ## hidden_outputs[inputs_to_neuron] * weights[inputs_to_neuron]

        total_output = outputs[connections[neuron.id]]
        weighted_output = np.dot(total_output, weights[neuron.id])

        neuron.input = weighted_output
        
        end_time = time()
        self.time_accumulated += (end_time-start_time)
        
    def changeWeights(self, guess, correct):

        eligible_hidden_neurons = np.where(self.hidden_layer_eligibility > 0.25)[0]
        weight_deltas = self.hidden_layer_eligibility[eligible_hidden_neurons] * LEARNING_RATE
        ## If incorrect, subtract weight_deltas, if correct, add.
        self.hidden_output_weights[guess][eligible_hidden_neurons] += (-(weight_deltas) + 2*correct*weight_deltas)

        ## change weights of elibigle input->hidden synapses
        for hidden_neuron in eligible_hidden_neurons:

            relevant_inputs = self.input_hidden_connections[hidden_neuron]
            relevant_eligibilities = self.input_layer_eligibility[relevant_inputs]

            eligible_input_indices = np.where(relevant_eligibilities > 0.25)[0]
            eligibile_input_neurons = relevant_inputs[eligible_input_indices]
            weight_deltas = self.input_layer_eligibility[eligibile_input_neurons] * LEARNING_RATE

            self.input_hidden_weights[hidden_neuron][eligible_input_indices] += (-(weight_deltas) + 2*correct*weight_deltas)

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

            if local_confidence > confidence:
                guess = num
                confidence = local_confidence
        #print(num_total_spikes)
        return guess
    
    def assignInputPixelValues(self, pixels):
        for num in range(len(self.input_layer_neurons)):

            neuron = self.input_layer_neurons[num]

            pixel_val_x = num % digit_drawing.NUM_PIXELS
            pixel_val_y = num // digit_drawing.NUM_PIXELS
            pixel = pixels[pixel_val_x][pixel_val_y]

            ## could potentially do this elsewhere since it only needs to be done once
            if pixel: ## colored black
                neuron.input = pixel / 255.0 

    def clearNeuronHistories(self):
        for neuron in self.ALL_NEURONS:
            neuron.reset()
    
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
            weights = np.random.uniform(0.05, 0.12, size=num_input_to_hidden)
            for _ in range(num_input_to_hidden): 
                connections.append(np.random.choice(self.input_layer_neurons).id)

            self.input_hidden_connections.append(np.array(connections))
            self.input_hidden_weights.append(weights)

        ## Output layer to hidden
        for neuron in self.output_layer_neurons:
            ## We're just connecting every hidden layer neuron to every output neuron
            connections = [_ for _ in range(NUM_HIDDEN_LAYER)]
            weights = np.random.uniform(0.125, 0.25, size=num_hidden_to_output)

            self.hidden_output_connections.append(np.array(connections))
            self.hidden_output_weights.append(weights)

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

    def run(self, images=[], labels=[], training=False):
        num_correct = 0
        ## TRAINING LOOP
        for img_num in range(len(images)):
            self.clearNeuronHistories()
            pixels = images[img_num]
            self.assignInputPixelValues(pixels)

            for timestep in range(NUM_TIMESTEPS):
                self.updateNeurons(timestep)

            guess = self.predictNumber()
            if guess == labels[img_num]:
                self.changeWeights(guess, correct=True)
                num_correct += 1
            elif guess == None:
                pass
            else:
                self.changeWeights(guess, correct=False)

            percent_correct = (float(num_correct)/(img_num+1))*100

            if not img_num % 100: ## every 100 images do an accuracy check
                print(f"Accuracy: {round(percent_correct,2)}")
        ## NETWORK DISPLAY FOR DEBUGGING
        # network_display = NetworkDisplay.NetworkDisplay() ## have to initialize this here otherwise it gets weird w/ DigitDraw
        # network_data = self.compileNetworkData()
        # network_display.run(network_data, NUM_TIMESTEPS)

        ## INPUT FOR TRAINED MODEL
        # digit_drawing.begin_drawing()

        # if digit_drawing.done:
        # else:
        #     print("No number drawn. Exiting...")
        #     sys.exit()

if __name__ == "__main__":
    ## MNIST data
    data_loader = MNISTLoader.MNISTDataloader(training_images_fp, training_labels_fp, test_images_fp, test_labels_fp)
    (training_images, training_labels), (test_images, test_labels) = data_loader.load_data()

    neural_net = LIFNeuralNetwork.loadConnectivity()
    if neural_net == None:
        neural_net = LIFNeuralNetwork()
        neural_net.connectNeurons()

    for epoch in range(NUM_EPOCHS):
        neural_net.run(training_images, training_labels)
        ## Shuffle images so it doesn't learn that pattern
        shuffled_indices = np.random.permutation(len(training_images))
        training_images = training_images[shuffled_indices]
        training_labels = training_labels[shuffled_indices]

