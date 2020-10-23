import player as p
import game as gm
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
                # 0 = empty, 1 = player one, 2 = player two, 3 = seed
                self.grid[row].append(0)

    def spawn_seed(self, player_one, player_two):
        # Spawn seed to a random position
        row = random.randint(0, gm.ROW_COUNT - 1)
        col = random.randint(0, gm.COLUMN_COUNT - 1)

        # Avoid spawning seed aligned to player
        if gm.TWO_PLAYERS:
            while row == player_one.row or row == player_two.row:
                row = random.randint(0, gm.ROW_COUNT - 1)
            while col == player_one.col or col == player_two.col:
                col = random.randint(0, gm.ROW_COUNT - 1)
        else:
            while row == player_one.row:
                row = random.randint(0, gm.ROW_COUNT - 1)
            while col == player_one.col:
                col = random.randint(0, gm.ROW_COUNT - 1)

        self.grid[row][col] = 3
        self.seed_pos = [row, col]