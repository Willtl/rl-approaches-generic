import sys, pygame
from pygame.locals import *
import random

import env.utils as utils
import env.game as gm


class Pygame:
    def __init__(self):
        # Pygame config
        pygame.init()
        self.size = width, height = gm.SCREEN_WIDTH, gm.SCREEN_HEIGHT
        self.screen = None
        self.row_count = gm.ROW_COUNT
        self.col_count = gm.COLUMN_COUNT

        # Game state and stuff
        self.game = gm.Game()

    def render(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode(self.size)

        self.screen.fill(utils.Color.BLACK.value)
        # Draw the grid
        for row in range(gm.ROW_COUNT):
            for column in range(gm.COLUMN_COUNT):
                # Figure out what color to draw the box
                if self.game.board.grid[row][column] == 3:
                    color = utils.Color.GREEN.value
                elif self.game.board.grid[row][column] == self.game.player_one.id:
                    color = utils.Color.BLUE.value
                elif self.game.player_two is not None and self.game.board.grid[row][column] == self.game.player_two.id:
                    color = utils.Color.RED.value
                else:
                    color = utils.Color.WHITE.value

                # Do the math to figure out where the center of the box is
                x = (gm.CELL_MARGIN + gm.WIDTH) * column + gm.CELL_MARGIN
                y = (gm.CELL_MARGIN + gm.HEIGHT) * row + gm.CELL_MARGIN

                # Draw the box
                pygame.draw.rect(self.screen, color, (x, y, gm.WIDTH, gm.HEIGHT), 0)

        pygame.display.flip()
        # pygame.display.update()
        # pygame.time.delay(10)


    def control(self, event):
        # Player one
        action = 0
        player = 0
        if event.key == K_w:
            action = 1
            player = 1
        elif event.key == K_d:
            action = 2
            player = 1
        elif event.key == K_s:
            action = 3
            player = 1
        elif event.key == K_a:
            action = 4
            player = 1
        self.game.step(action, player)

        if gm.TWO_PLAYERS:
            action = 0
            player = 0
            if event.key == K_UP:
                action = 1
                player = 2
            elif event.key == K_RIGHT:
                action = 2
                player = 2
            elif event.key == K_DOWN:
                action = 3
                player = 2
            elif event.key == K_LEFT:
                action = 4
                player = 2
            self.game.step(action, player)

    def pump(self):
        pygame.event.pump()

    def run(self):
        quit = False
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    quit = True
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    quit = True

                if event.type == pygame.KEYDOWN:
                    self.control(event)

            if quit:
                return

            self.render()


if __name__ == "__main__":
    main()