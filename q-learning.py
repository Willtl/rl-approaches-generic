import env.gridseedpygame as env
import numpy as np
import time

'''
The game support multiple seeds and players. In this example, I am considering only one player, 
4 rows and 4 cols, and one seed 
'''

# Q-table size
q_size = [env.gm.ROW_COUNT, env.gm.COLUMN_COUNT, env.gm.ROW_COUNT, env.gm.COLUMN_COUNT, 4]
q_table = np.random.uniform(low=0, high=0, size=q_size)

# Q-learning parameters
learning_rate = 0.1
discount = 0.95
episodes = 100000
zero_eps_at = int(0.8 * episodes)


def get_state(state):
    seed_pos = state.seed_pos
    one_pos = state.one_pos
    return tuple([seed_pos[0], seed_pos[1], one_pos[0], one_pos[1]])


if __name__ == '__main__':
    game_instance = env.Pygame()
    # game_instance.run()

    # Initial state
    matrix_state = get_state(game_instance.game.reset())

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
        new_state, reward = game_instance.game.step(action + 1, 1)    # reward 0 if got seed, -1 otherwise
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
            game_instance.pump()
            game_instance.render()
            time.sleep(0.1)

        cur_it += 1
