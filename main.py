import DigitDraw
import LIFNeuron
import numpy as np
import pickle
import threading

FILEPATH = "C:/Users/jacob/OneDrive/Desktop/Code/Projects/DigitRec Neural Network/"
NUM_INPUT_LAYER = 784
NUM_HIDDEN_LAYER = 200
NUM_OUTPUT_LAYER = 10

digit_drawing = DigitDraw.DigitDraw()
running = True

input_layer_neurons = np.array([LIFNeuron.LIF_Neuron(type=0) for _ in range(NUM_INPUT_LAYER)])
hidden_layer_neurons = np.array([LIFNeuron.LIF_Neuron(type=1) for _ in range(NUM_HIDDEN_LAYER)])
output_layer_neurons = np.array([LIFNeuron.LIF_Neuron(type=2) for _ in range(NUM_OUTPUT_LAYER)])

input_hidden_connections = {} ## LIFNeuron: [LIFNeuron, LIFNeuron, LIFNeuron, etc] -> the key receives input from the list of neurons
input_hidden_weights = {} ## {LIFNeuron: [0.1,0.2,0.3,0.4]} etc

hidden_output_connections = {}
hidden_output_weights = {} 

def updateNeurons():
    pixels = digit_drawing.pixels
    for num in range(NUM_INPUT_LAYER): ## probably numpy vectorize this
        neuron = input_layer_neurons[num]
        neuron.input = 0 ## reset

        pixel_val_x = num % 28
        pixel_val_y = num // 28
        pixel = pixels[pixel_val_x][pixel_val_y]

        if pixel: ## colored black
           neuron.input = 1

def connectNeurons():
    num_input_to_hidden = int(np.random.uniform(0.1, 0.3) * NUM_HIDDEN_LAYER)
    num_hidden_to_output = NUM_OUTPUT_LAYER ## since there's only 10 output neurons, maybe change this

    for neuron in hidden_layer_neurons:
        connections = []
        weights = np.random.uniform(0.01, 0.05, size=num_input_to_hidden)
        for _ in range(num_input_to_hidden): 
            connections.append(np.random.choice(input_layer_neurons))

        input_hidden_connections[neuron] = connections
        input_hidden_weights[neuron] = weights

    for neuron in output_layer_neurons:
        connections = []
        weights = np.random.uniform(0.01, 0.05, size=num_hidden_to_output)
        for _ in range(num_hidden_to_output): 
            connections.append(np.random.choice(hidden_layer_neurons))

        hidden_output_connections[neuron] = connections
        hidden_output_weights[neuron] = weights

    saveConnectivity()

def loadConnectivity():
    try:
        data = None
        with open(f"{FILEPATH}/data/connections.pkl", "rb") as f:
            data = pickle.load(f)
        input_hidden_connections.update(data["input_hidden_connections"])
        input_hidden_weights.update(data["input_hidden_weights"])
        hidden_output_connections.update(data["hidden_output_connections"])
        hidden_output_weights.update(data["hidden_output_weights"])
        print("Data imported!")
    except:
        print("Import failed. Generate new connectivity? Y/N")
        if input("") == 'Y':
            connectNeurons()
        else:
            return

def saveConnectivity():
    with open(f"{FILEPATH}/data/connections.pkl", "wb") as f:
        pickle.dump({
            "input_hidden_connections": input_hidden_connections,
            "input_hidden_weights": input_hidden_weights,
            "hidden_output_connections": hidden_output_connections,
            "hidden_output_weights": hidden_output_weights
        }, f)

def run():
    while running:
        updateNeurons()

def draw():
    digit_drawing.begin_drawing()

if __name__ == "__main__":
    loadConnectivity()
    thread = threading.Thread(target=run)
    thread.start()
    digit_drawing.begin_drawing()
