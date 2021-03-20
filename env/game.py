import env.board as board
import env.player as player
import random

# Game configuration
NUMB_SEEDS = 1
SEED_SPAWN_TIME = 100  # ms

# Number of rows and column in the grid
ROW_COUNT = 5
COLUMN_COUNT = 5

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


class State:
    def __init__(self, _seed_pos, _one_pos):
        self.seed_pos = _seed_pos
        self.one_pos = _one_pos


class Game:
    def __init__(self):
        self.board = board.Board()
        self.player_one = player.Player(1)
        # Spawn players
        self.spawn_player(self.player_one)
        # Spawn seeds
        for i in range(NUMB_SEEDS):
            self.board.spawn_seed(self.player_one)

    def step(self, action):
        reward = 0
        if action == 1:
            reward = self.move_player(self.player_one, -1, 0)
        if action == 2:
            reward = self.move_player(self.player_one, 0, 1)
        if action == 3:
            reward = self.move_player(self.player_one, 1, 0)
        if action == 4:
            reward = self.move_player(self.player_one, 0, -1)

        # Array with two values with the seed position
        seed_pos = self.board.seed_pos
        # Player one position
        one_pos = [self.player_one.row, self.player_one.col]
        state = State(seed_pos, one_pos)

        return state, reward

    def spawn_player(self, player):
        # Spawn player to a random position
        row = random.randint(0, ROW_COUNT - 1)
        col = random.randint(0, COLUMN_COUNT - 1)

        player.row += row
        player.col += col

        # Update grid with new position
        self.board.grid[player.row][player.col] = player.id

    def move_player(self, player, row, col):
        reward = 0
        new_row = player.row + row
        new_col = player.col + col

        if new_row < 0 or new_row > ROW_COUNT - 1 or new_col < 0 or new_col > COLUMN_COUNT - 1:
            reward = None
        else:
            # Remove player from current position on the board
            self.board.grid[player.row][player.col] = 0
            # New position
            player.row = new_row
            player.col = new_col
            # Check if new position increase score
            if self.board.grid[player.row][player.col] == 3:
                player.score += 1.0
                reward = 1.0
                self.board.spawn_seed(self.player_one)
            # else:     # used only on q-learning
            #    reward = -0.01    # used only on q-learning
            # Update grid with new position
            self.board.grid[player.row][player.col] = player.id

        return reward

    def reset(self):
        self.board = board.Board()
        self.player_one = player.Player(1)
        self.spawn_player(self.player_one)

        # Spawn seeds
        for i in range(NUMB_SEEDS):
            self.board.spawn_seed(self.player_one)

        # Array with two values with the seed position
        seed_pos = self.board.seed_pos
        # Player one position
        one_pos = [self.player_one.row, self.player_one.col]
        state = State(seed_pos, one_pos)
        return state