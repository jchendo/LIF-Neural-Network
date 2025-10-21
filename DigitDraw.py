import pygame as pg
import numpy as np

pg.init()
screen_size = 560

class DigitDraw:

    pixels = np.zeros(shape=(28,28), dtype=bool)
    screen = pg.display.set_mode((screen_size,screen_size))
    screen.fill("white")
    running = True
    done = False

    def __init__(self):
        pass

    def clear_screen(self):
        self.screen.fill("white")

    def color_square(self,mouse_pos):
        grid_x = int((mouse_pos[0] / screen_size) * 28)
        grid_y = int((mouse_pos[1] / screen_size) * 28)

        self.pixels[grid_x][grid_y] = 1

        pg.draw.rect(self.screen, "black", pg.Rect(grid_x*20, grid_y*20, 20, 20))

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

