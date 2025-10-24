import pygame as pg
import time

class NetworkDisplay:

    input_layer_circles = []
    hidden_layer_circles = []
    output_layer_circles = []
    num_outputs = 2 ## potentially temporary just for testing
    num_inputs = 4
    running = True
    data = None
    timesteps = 0
    ## open window etc etc
    screen_size = (1080, 720)
    last_tick = pg.time.get_ticks() 


    def __init__(self):
        pg.init()
        pg.font.init()
        pg.display.set_caption('Neural Network Display')
        ## weird
        self.screen = pg.display.set_mode((self.screen_size[0],self.screen_size[1]))
        self.screen.fill("white")
        self.font = pg.font.SysFont('Arial', 24)
        

    ## hidden connections takes a neuron as a key input & returns a list of all of the neurons it receives input from
    ## see main.py
    def display_network_subset(self, data, num_timesteps): ## remove the defaults later

        input_neurons, hidden_neurons, output_neurons, input_hidden_connections = data

        hidden_offset = self.screen_size[1] / (len(input_hidden_connections.keys()) + 1)
        output_offset = self.screen_size[1] / (self.num_outputs + 1)
        input_offset = self.screen_size[1] / (self.num_inputs + 1)
        
        clock = pg.time.Clock()

        timestep = 0
        while timestep < num_timesteps: ## holy for loop
            self.handleEvents()
            now = pg.time.get_ticks()

            if now - self.last_tick > 500: ## pygame doesn't play well with time.sleep() or delays, so this is a workaround
                self.screen.fill("white")
                self.last_tick = now

                ## input layer
                for num in range(self.num_inputs):
                    voltage = round(input_neurons[num].voltages[timestep],2)
                    coords = (200, input_offset*(num+1))
                    self.input_layer_circles.append(coords) ## change this to actually be used

                    ## render voltage text above each neuron
                    text = self.font.render(str(voltage), True, "black")
                    self.screen.blit(text, (coords[0], coords[1]+30))

                    if input_neurons[num].spikes[timestep] != 1:
                        pg.draw.circle(self.screen, "black", center=coords, radius=20, width=1) # input neuron
                    else:
                        pg.draw.circle(self.screen, "black", center=coords, radius=20)

                ## hidden layer
                for num in range(len(input_hidden_connections.keys())):
                    connections = input_hidden_connections[num]
                    voltage = round(hidden_neurons[num].voltages[timestep],2)
                    coords = (450, hidden_offset*(num+1))
                    self.hidden_layer_circles.append(coords)

                    text = self.font.render(str(voltage), True, "black")
                    self.screen.blit(text, (coords[0], coords[1]+30))

                    if hidden_neurons[num].spikes[timestep] != 1:
                        pg.draw.circle(self.screen, "black", center=coords, radius=8, width=1)
                    else:
                        pg.draw.circle(self.screen, "black", center=coords, radius=8)
                    
                    for connection in connections:
                        pg.draw.line(self.screen, "black", coords, self.input_layer_circles[connection])

                ## output layer
                for num in range(self.num_outputs): ## outputs
                    voltage = round(output_neurons[num].voltages[timestep],2)
                    coords = (825, output_offset*(num+1))
                    self.output_layer_circles.append(coords)
                    

                    text = self.font.render(str(voltage), True, "black")
                    self.screen.blit(text, (coords[0], coords[1]+30))

                    if output_neurons[num].spikes[timestep] != 1:
                        pg.draw.circle(self.screen, "black", center=coords, radius=25, width=1)
                    else:
                        pg.draw.circle(self.screen, "black", center=coords, radius=25)

                    for hidden_coord in self.hidden_layer_circles: ## connect every hidden neuron to every output
                        pg.draw.line(self.screen, "black", hidden_coord, coords)

                timestep += 1
                
            pg.display.flip()
            clock.tick(60)

    def handleEvents(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False

            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_RIGHT:
                    print(event)
                    self.display_network_subset(self.data, self.timesteps)

    def run(self, data, timesteps):
        self.data = data
        self.timesteps = timesteps
        while self.running:
            self.handleEvents()

            pg.display.flip()

