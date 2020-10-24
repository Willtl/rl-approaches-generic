import env.gridseedpygame as env
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
The game support multiple seeds and players. In this example, I am considering only one player, 
4 rows and 4 cols, and one seed 
'''

# Neural network parameters
learning_rate = 0.001
# How many change of states should be used to train the model
train_iterations = 2000000
# Once the model is trained, how many iterations should be rendered to show de results
test_iterations = 2000

# Q-table size
q_size = [env.gm.ROW_COUNT, env.gm.COLUMN_COUNT, env.gm.ROW_COUNT, env.gm.COLUMN_COUNT, 4]
q_table = np.random.uniform(low=0, high=0, size=q_size)

# Q-learning parameters
discount = 0.95


class ANN(nn.Module):
    # ANN's layer architecture
    def __init__(self):
        # Initialize superclass
        super().__init__()
        # Fully connected layers
        self.inputs = 4
        self.outputs = 4
        self.l1 = nn.Linear(self.inputs, 6)  # To disable bias use bias=False
        self.l2 = nn.Linear(6, 5)
        self.l3 = nn.Linear(5, 4)
        self.l4 = nn.Linear(4, self.outputs)

    # Define how the data passes through the layers
    def foward(self, x):
        # Passes x through layer one and activate with rectified linear unit function
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        # Linear output layer
        x = self.l4(x)
        return x

    def feed(self, x):
        # x = torch.rand((28, 28))
        # x = x.view(1, 28 * 28)
        outputs = self.foward(x)
        return outputs

    def train_net(self, trainset):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        epochs = 10
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in trainset:
                # the inputs
                x, y = batch
                # zero the parameter gradients
                self.zero_grad()
                # Foward
                outputs = self.feed(x.view(-1, 28 * 28))
                # For [0, 1, 0, 0] vectors, use mean squared error, for scalar values use nll_loss
                loss = F.nll_loss(outputs, y)
                # Back propagate the loss
                loss.backward()
                # Adjust the weights
                optimizer.step()
                # Calculate epoch loss
                epoch_loss += outputs.shape[0] * loss.item()
            print("Epoch loss: ", epoch_loss / len(trainset))

    def test_net(self, testset):
        # Deactivate Dropout and BatchNorm
        self.eval()

        correct = 0
        total = 0
        # Deactivate gradient calculations
        with torch.no_grad():
            # Check each batch in testset
            for batch in testset:
                x, y, = batch
                outputs = self.feed(x.view(-1, 28 * 28))
                # Loop through outputs and check if it is correct or not
                for idx, i in enumerate(outputs):
                    if torch.argmax(i) == y[idx]:
                        correct += 1
                    total += 1
            print("Accuracy: ", round(correct/total, 3))

        # Activate Dropout and BatchNorm again
        self.train()

    def update_weights(self):
        print(self.l1.weight)
        for i in range(64):
            print(self.l1.weight[i])
        # for j in range(28 * 28):
        #     print(self.l1.weight[i][j])
        weight = nn.Parameter(torch.ones_like(self.l1.weight))
        print(weight)

    def print(self):
        print(self)


def get_state(state):
    seed_pos = state.seed_pos
    one_pos = state.one_pos
    return tuple([seed_pos[0], seed_pos[1], one_pos[0], one_pos[1]])


if __name__ == '__main__':
    model = ANN()
    print(model)
    x = torch.rand((2, 2))
    print(x)
    x[0][0] = 0
    print(x[0][0])
    x = x.view(-1, 4)
    print(x)

    output = model.feed(x)
    print(f"output {output}")
    '''
    # Initial state
    state = get_state(env.game.reset())
    
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
        new_state, reward = env.game.step(action + 1, 1)    # reward 0 if got seed, -1 otherwise
        new_matrix_state = get_state(new_state)

        # Q-learning equation
        max_future_q = np.max(q_table[new_matrix_state])
        current_q = q_table[matrix_state][action]
        new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
        q_table[matrix_state][action] = new_q

        # Update state
        matrix_state = new_matrix_state

        if cur_it + 2000 >= episodes:
            # Pump events
            env.pygame.event.pump()
            env.render()
            time.sleep(0.1)

        cur_it += 1
    '''
