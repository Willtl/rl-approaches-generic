import env.gridseedpygame as env
import numpy as np
import time

# The game support multiple seeds and players. In this example, I am considering only one player, 4 rows and 4 cols, and one seed

# Number of states = n * n - 1, where n are the number of cells in the grid
n_states = (env.gm.ROW_COUNT * env.gm.COLUMN_COUNT) * ((env.gm.ROW_COUNT * env.gm.COLUMN_COUNT) - env.gm.NUMB_SEEDS)
n_actions = 4
os_size = [env.gm.ROW_COUNT, env.gm.COLUMN_COUNT, env.gm.ROW_COUNT, env.gm.COLUMN_COUNT, 4]

# Q-learning parameters
q_table = np.random.uniform(low=0, high=0, size=os_size)
learning_rate = 0.1
discount = 0.95
episodes = 2000000
zero_eps_at = int(0.8 * episodes)


def get_state(state):
    # print(state)
    # Used to retrieval the of the states where seed is at seed_pos[0] and seed_pos[1]
    seed_pos = env.game.reset().seed_pos
    # Used to retrieval of the actions' reward
    one_pos = env.game.reset().one_pos
    return tuple([seed_pos[0], seed_pos[1], one_pos[0], one_pos[1]])


if __name__ == '__main__':
    print(f"Number of possible states: {n_states}")

    state = env.game.reset()
    matrix_state = get_state(state)

    print(matrix_state)
    print(q_table[matrix_state])
    print(np.argmax(q_table[matrix_state]))
    print(q_table[matrix_state][np.argmax(q_table[matrix_state])])

    cur_it = 0
    while cur_it < episodes:
        if cur_it % 10000 == 0:
            print(cur_it)

        # Linearly decrease epsilon
        epsilon = 0.5 - cur_it * (0.5 / zero_eps_at)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[matrix_state])
        else:
            action = np.random.randint(0, 4)

        # Update game with action, player one
        new_state, reward = env.game.step(action + 1, 1)
        new_matrix_state = get_state(new_state)

        # Q-learning equation
        max_future_q = np.max(q_table[new_matrix_state])
        current_q = q_table[matrix_state][action]
        new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
        q_table[matrix_state][action] = new_q

        # Update state
        matrix_state = new_matrix_state

        if cur_it > 0.99 * episodes:
            # Pump events
            env.pygame.event.pump()
            env.render()
            time.sleep(0.1)

        cur_it += 1
