import sys, pygame
from pygame.locals import *
import random

import utils
import game as gm

# Pygame config
pygame.init()
size = width, height = gm.SCREEN_WIDTH, gm.SCREEN_HEIGHT
screen = pygame.display.set_mode(size)

# Game state and stuff
game = gm.Game()
RENDER = True


def render():
    screen.fill(utils.Color.BLACK.value)
    # Draw the grid
    for row in range(gm.ROW_COUNT):
        for column in range(gm.COLUMN_COUNT):
            # Figure out what color to draw the box
            if game.board.grid[row][column] == 3:
                color = utils.Color.GREEN.value
            elif game.board.grid[row][column] == game.player_one.id:
                color = utils.Color.BLUE.value
            elif game.player_two is not None and game.board.grid[row][column] == game.player_two.id:
                color = utils.Color.RED.value
            else:
                color = utils.Color.WHITE.value

            # Do the math to figure out where the center of the box is
            x = (gm.CELL_MARGIN + gm.WIDTH) * column + gm.CELL_MARGIN
            y = (gm.CELL_MARGIN + gm.HEIGHT) * row + gm.CELL_MARGIN

            # Draw the box
            pygame.draw.rect(screen, color, (x, y, gm.WIDTH, gm.HEIGHT), 0)

    # pygame.display.flip()
    pygame.display.update()
    # pygame.time.delay(10)


def control(event):
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
    game.step(action, player)

    # Player two
    action = 0
    player = 0
    if gm.TWO_PLAYERS:
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
    game.step(action, player)


def main():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                control(event)

        if RENDER:
            render()


if __name__ == "__main__":
    main()