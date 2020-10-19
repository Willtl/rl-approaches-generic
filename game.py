import board
import player
import random

# Game configuration
NUMB_SEEDS = 2
SEED_SPAWN_TIME = 100  # ms
TWO_PLAYERS = True

# How many rows and columns we will have
ROW_COUNT = 10
COLUMN_COUNT = 17

# WIDTH and HEIGHT of each grid location
WIDTH = 30
HEIGHT = 30

# Margin between each cell and on the edges of the screen.
CELL_MARGIN = 5

# Top margin
TOP_MARGIN = CELL_MARGIN * 2 + HEIGHT

# Do the math to figure out our screen dimensions
SCREEN_WIDTH = (WIDTH + CELL_MARGIN) * COLUMN_COUNT + CELL_MARGIN
SCREEN_HEIGHT = (HEIGHT + CELL_MARGIN) * ROW_COUNT + CELL_MARGIN + TOP_MARGIN


class Game:
    def __init__(self):
        self.board = board.Board()
        self.player_one = player.Player(1)
        self.player_two = None
        if TWO_PLAYERS:
            self.player_two = player.Player(2)
        # Spawn players
        self.spawn_player(self.player_one)
        if TWO_PLAYERS:
            self.spawn_player(self.player_two)
        # Spawn seeds
        for i in range(NUMB_SEEDS):
            self.board.spawn_seed(self.player_one, self.player_two)

    def step(self, action, player):
        if player == 1:
            if action == 1 and self.player_one.row > 0:
                self.move_player(self.player_one, -1, 0)
            if action == 2 and self.player_one.col < COLUMN_COUNT - 1:
                self.move_player(self.player_one, 0, 1)
            if action == 3 and self.player_one.row < ROW_COUNT - 1:
                self.move_player(self.player_one, 1, 0)
            if action == 4 and self.player_one.col > 0:
                self.move_player(self.player_one, 0, -1)

        if player == 2:
            if action == 1 and self.player_two.row > 0:
                self.move_player(self.player_two, -1, 0)
            if action == 2 and self.player_two.col < COLUMN_COUNT - 1:
                self.move_player(self.player_two, 0, 1)
            if action == 3 and self.player_two.row < ROW_COUNT - 1:
                self.move_player(self.player_two, 1, 0)
            if action == 4 and self.player_two.col > 0:
                self.move_player(self.player_two, 0, -1)

    def spawn_player(self, player):
        # Spawn player to a random position
        row = random.randint(0, ROW_COUNT - 1)
        col = random.randint(0, COLUMN_COUNT - 1)
        row = 2
        col = 0
        the_other_player = None
        if TWO_PLAYERS:
            if self.player_one == player:
                the_other_player = self.player_two
            else:
                the_other_player = self.player_one

            # Make sure it is not at same row/col of second player
            while the_other_player.row == row:
                row = random.randint(0, ROW_COUNT - 1)
            while the_other_player.col == col:
                col = random.randint(0, COLUMN_COUNT - 1)

        # Update player's position given new position
        self.move_player(player, row, col)
        
    def move_player(self, player, row, col):
        if TWO_PLAYERS:
            the_other_player = None
            if self.player_one == player:
                the_other_player = self.player_two
            else:
                the_other_player = self.player_one

            # Set current cell to empty if both player are not in the same position
            if player.row == the_other_player.row and player.col == the_other_player.col:
                self.board.grid[player.row][player.col] = the_other_player.id
            else:
                self.board.grid[player.row][player.col] = 0
        else:
            self.board.grid[player.row][player.col] = 0

        # New position
        player.row += row
        player.col += col

        # Check if new position increase score
        if self.board.grid[player.row][player.col] == 3:
            player.score += 1
            self.board.spawn_seed(self.player_one, self.player_two)
            print(f"Score Player 1: {self.player_one.score}, Player 2: {self.player_two.score}")

        # Update grid with new position
        self.board.grid[player.row][player.col] = player.id