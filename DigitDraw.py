import pygame as pg
import numpy as np

screen_size = 560

class DigitDraw:

    NUM_PIXELS = 2
    pixels = np.zeros(shape=(NUM_PIXELS,NUM_PIXELS), dtype=bool)
    screen = pg.display.set_mode((screen_size,screen_size))
    screen.fill("white")
    running = True
    done = False

    def __init__(self):
        pg.init()
        pg.display.set_caption('Draw a Number - C to clear, Enter to confirm')

    def clear_screen(self):
        self.screen.fill("white")

    def color_square(self,mouse_pos):
        grid_x = int((mouse_pos[0] / screen_size) * self.NUM_PIXELS)
        grid_y = int((mouse_pos[1] / screen_size) * self.NUM_PIXELS)

        adjacent_pixels = []
        #adjacent_pixels = [(grid_x+1, grid_y), (grid_x-1, grid_y), (grid_x, grid_y+1), (grid_x, grid_y-1)]
        pixel_offset = screen_size / self.NUM_PIXELS ## size of each grid square

        ## maybe clean this up
        for pixel in adjacent_pixels:
            if pixel[0] < self.NUM_PIXELS and pixel[1] < self.NUM_PIXELS:
                pg.draw.rect(self.screen, "black", pg.Rect(pixel[0]*pixel_offset, pixel[1]*pixel_offset, pixel_offset, pixel_offset))
                self.pixels[pixel[0]][pixel[1]] = 1

        if grid_x < self.NUM_PIXELS and grid_y < self.NUM_PIXELS:
            self.pixels[grid_x][grid_y] = 1
            pg.draw.rect(self.screen, "black", pg.Rect(grid_x*pixel_offset, grid_y*pixel_offset, pixel_offset, pixel_offset))

    def begin_drawing(self):
        while self.running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.running = False
                elif event.type == pg.MOUSEBUTTONDOWN:
                    self.color_square(pg.mouse.get_pos())  # draw while dragging
                elif event.type == pg.MOUSEMOTION and pg.mouse.get_pressed()[0]:
                    self.color_square(pg.mouse.get_pos())  # draw while dragging
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_c:
                        self.clear_screen()
                    if event.key == pg.K_RETURN:
                        self.running = False
                        self.done = True
                        pg.quit()
                        return

            pg.display.flip()