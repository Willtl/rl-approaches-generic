import env.player as player
import env.game as gm
import random


class Board:
    def __init__(self):
        #  2-dimensional array
        self.grid = []
        self.seed_pos = []

        for row in range(gm.ROW_COUNT):
            self.grid.append([])
            for column in range(gm.COLUMN_COUNT):
                # Values in cells are equivalent to
                self.grid[row].append(0)

    def spawn_seed(self, player_one):
        # Spawn seed to a random position
        row = random.randint(0, gm.ROW_COUNT - 1)
        col = random.randint(0, gm.COLUMN_COUNT - 1)

        # Avoid spawning seed aligned to player
        while row == player_one.row:
            row = random.randint(0, gm.ROW_COUNT - 1)
        while col == player_one.col:
            col = random.randint(0, gm.ROW_COUNT - 1)

        self.grid[row][col] = 3
        self.seed_pos = [row, col]