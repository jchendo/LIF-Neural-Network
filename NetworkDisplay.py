import pygame as pg

pg.init()
pg.font.init()
pg.display.set_caption('Neural Network Display')
screen_size = (1080, 720)

class NetworkDisplay:

    input_layer_circles = []
    hidden_layer_circles = []
    output_layer_circles = []

    screen = pg.display.set_mode((screen_size[0],screen_size[1]))
    screen.fill("white")
    font = pg.font.SysFont('Arial', 12)
    text_surface = font.render('0', True, "black")
    running = True

    def display_network_subset(self, neuron=0, hidden_connections=[0]*40, input_hidden_weights=[], hidden_output_weights=[]): ## remove the defaults later
        #voltages = [str(next.Vm) for next in connections]
        input_circle_coords = (100, (screen_size[1]/2)-10)
        self.input_layer_circles = [input_circle_coords] ## keep track of coords and redraw when spiking
        
        pg.draw.circle(self.screen, "black", center=input_circle_coords, radius=20) # input neuron

        for num in range(len(hidden_connections)):
            coords = (450, 17.5 * (num+1))
            self.hidden_layer_circles.append(coords)

            pg.draw.line(self.screen, "black", input_circle_coords, coords)
            pg.draw.circle(self.screen, "black", center=coords, radius=8)
            # self.text_surface = self.font.render(voltages[num], True, "black")
            # self.screen.blit(self.text_surface, coords)

        for num in range(10): ## ten outputs
            coords = (825, 65*(num+1))
            self.output_layer_circles.append(coords)
            pg.draw.circle(self.screen, "black", center=coords, radius=10)

            for hidden_coord in self.hidden_layer_circles:
                pg.draw.line(self.screen, "black", hidden_coord, coords)

    def run(self):
        while self.running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False

                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_RIGHT:
                        self.display_network_subset()

            pg.display.flip()




display = NetworkDisplay()
display.run()


