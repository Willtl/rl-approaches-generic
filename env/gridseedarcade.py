import arcade
from pyglet import image
import random


# Game configuration
NUMB_SEEDS = 2
SEED_SPAWN_TIME = 100  # ms
TWO_PLAYERS = True

# Render the game if true
RENDER = True

# Frame rate
FRAME_RATE = 60

# How many rows and columns we will have
ROW_COUNT = 10
COLUMN_COUNT = 10

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
SCREEN_TITLE = "Seed Game"


class Player:
    def __init__(self, _id):
        self.id = _id
        self.row = 0
        self.col = 0
        self.score = 0

    def draw_score(self):
        if self.id == 1:
            start_x = CELL_MARGIN
            start_y = (CELL_MARGIN + HEIGHT) * ROW_COUNT + CELL_MARGIN + HEIGHT // 2
            # arcade.draw_point(start_x, start_y, arcade.color.BLUE, 5)
            text = arcade.draw_text(f"Player {self.id}: {self.score}", start_x, start_y, arcade.color.WHITE, 12,
                                    anchor_x="left", anchor_y="center")
            arcade.draw_line(start_x, start_y - (text.height // 2), start_x + text.width, start_y - (text.height // 2),
                             arcade.color.BLUE)
        else:
            start_x = (WIDTH + CELL_MARGIN) * COLUMN_COUNT
            start_y = (CELL_MARGIN + HEIGHT) * ROW_COUNT + CELL_MARGIN + HEIGHT // 2
            # arcade.draw_point(start_x, start_y, arcade.color.RED, 5)
            text = arcade.draw_text(f"Player {self.id}: {self.score}", start_x, start_y, arcade.color.WHITE, 12,
                                    anchor_x="right", anchor_y="center")
            arcade.draw_line(start_x, start_y - (text.height // 2), start_x - text.width, start_y - (text.height // 2),
                             arcade.color.RED)


class GridGame(arcade.Window):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)

        #  2-dimensional array
        self.grid = []
        for row in range(ROW_COUNT):
            self.grid.append([])
            for column in range(COLUMN_COUNT):
                # Values in cells are equivalent to
                # 0 = empty, 1 = player one, 2 = player two, 3 = seed
                self.grid[row].append(0)

        # Instantiate
        self.player_one = Player(1)
        self.player_two = None
        if TWO_PLAYERS:
            self.player_two = Player(2)

        # Spawn players
        self.spawn_player(self.player_one)
        if TWO_PLAYERS:
            self.spawn_player(self.player_two)

        # Spawn first seeds
        for i in range(NUMB_SEEDS):
            self.spawn_seed()

        # Set background to black
        arcade.set_background_color(arcade.color.BLACK)

    def step(self, action):
        if random.random() < 0.001:
            return True
        else:
            return False

    def on_draw(self):
        if RENDER:
            # This command has to happen before we start drawing
            arcade.start_render()

            # Draw the grid
            for row in range(ROW_COUNT):
                for column in range(COLUMN_COUNT):
                    # Figure out what color to draw the box
                    if self.grid[row][column] == 3:
                        color = arcade.color.GREEN
                    elif self.grid[row][column] == self.player_one.id:
                        color = arcade.color.BLUE
                    elif self.player_two is not None and self.grid[row][column] == self.player_two.id:
                        color = arcade.color.RED
                    else:
                        color = arcade.color.WHITE

                    # Do the math to figure out where the center of the box is
                    x = (CELL_MARGIN + WIDTH) * column + CELL_MARGIN + WIDTH // 2
                    y = (CELL_MARGIN + HEIGHT) * row + CELL_MARGIN + HEIGHT // 2

                    # Draw the box
                    arcade.draw_rectangle_filled(x, y, WIDTH, HEIGHT, color)

            self.player_one.draw_score()
            if TWO_PLAYERS:
                self.player_two.draw_score()

    def on_key_press(self, key, modifiers):
        # Player one controls
        if key == arcade.key.W:
            if self.player_one.row < ROW_COUNT - 1:
                self.move_player(self.player_one, 1, 0)
        elif key == arcade.key.D:
            if self.player_one.col < COLUMN_COUNT - 1:
                self.move_player(self.player_one, 0, 1)
        elif key == arcade.key.S:
            if self.player_one.row > 0:
                self.move_player(self.player_one, -1, 0)
        elif key == arcade.key.A:
            if self.player_one.col > 0:
                self.move_player(self.player_one, 0, -1)
        print(f"Player's {self.player_one.id}, position ({self.player_one.row}, "
              f"{self.player_one.col}), score: {self.player_one.score}")

        # Player two controls
        if TWO_PLAYERS:
            if key == arcade.key.UP:
                if self.player_two.row < ROW_COUNT - 1:
                    self.move_player(self.player_two, 1, 0)
            elif key == arcade.key.RIGHT:
                if self.player_two.col < COLUMN_COUNT - 1:
                    self.move_player(self.player_two, 0, 1)
            elif key == arcade.key.DOWN:
                if self.player_two.row > 0:
                    self.move_player(self.player_two, -1, 0)
            elif key == arcade.key.LEFT:
                if self.player_two.col > 0:
                    self.move_player(self.player_two, 0, -1)

            print(f"Player's {self.player_two.id} ,position ({self.player_two.row}, "
                  f"{self.player_two.col}), score: {self.player_two.score}", end="\n\n")

    def move_player(self, player, row, col):
        if TWO_PLAYERS:
            the_other_player = None
            if self.player_one == player:
                the_other_player = self.player_two
            else:
                the_other_player = self.player_one

            # Set current cell to empty if both player are not in the same position
            if player.row == the_other_player.row and player.col == the_other_player.col:
                self.grid[player.row][player.col] = the_other_player.id
            else:
                self.grid[player.row][player.col] = 0
        else:
            self.grid[player.row][player.col] = 0

        # New position
        player.row += row
        player.col += col

        # Check if new position increase score
        if self.grid[player.row][player.col] == 3:
            player.score += 1
            self.spawn_seed()

        # Update grid with new position
        self.grid[player.row][player.col] = player.id

    def move_player_restrictive(self, player, row, col):
        the_other_player = None
        if self.player_one == player:
            the_other_player = self.player_two
        else:
            the_other_player = self.player_one

        # Players cant be at same cell at same time
        new_row = player.row + row
        new_col = player.col + col

        if the_other_player.row != new_row or the_other_player.col != new_col:
            # Remove player from current position in the grid
            self.grid[player.row][player.col] = 0

            # New position
            player.row += row
            player.col += col

            # Update grid with new position
            self.grid[player.row][player.col] = player.id

    def spawn_player(self, player):
        # Spawn player to a random position
        row = random.randint(0, ROW_COUNT - 1)
        col = random.randint(0, COLUMN_COUNT - 1)

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

    def spawn_seed(self):
        # Spawn seed to a random position
        row = random.randint(0, ROW_COUNT - 1)
        col = random.randint(0, COLUMN_COUNT - 1)

        # Avoid spawning seed aligned to player
        if TWO_PLAYERS:
            while row == self.player_one.row or row == self.player_two.row:
                row = random.randint(0, ROW_COUNT - 1)
            while col == self.player_one.col or col == self.player_two.col:
                col = random.randint(0, ROW_COUNT - 1)

        self.grid[row][col] = 3


def main():
    game = GridGame(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    # Set max frame rate
    arcade.application.Window.set_update_rate(game, 0.0)
    # Set icon
    icon = image.load('images/icon.png')
    arcade.application.Window.set_icon(game, icon)

    # # Main loop
    # done = False
    # while not done:
    #     action = 1
    #     new_state, reward, done = game.step(action)    # update game state
    #     game.reder()    # render new state

    # Run the game
    arcade.run()


if __name__ == "__main__":
    main()