import env.gridseedpygame as env
import numpy as np
import random
import time
import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

'''
The game support multiple seeds and players. In this example, I am considering only one player, 
4 rows and 4 cols, and one seed 
'''

# How many change of states should be used to train the model
max_epochs = 100
# How many moves should be stored and size of batch to back propagate
n_samples = 500
batch_size = 25
# Once the model is trained, how many iterations should be rendered to show de results
test_iterations = 2000

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
        self.l1 = nn.Linear(self.inputs, 4)  # To disable bias use bias=False
        self.l2 = nn.Linear(4, 4)
        self.l3 = nn.Linear(4, 4)
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
        outputs = self.foward(x)
        return outputs

    def backward(self, dataloader):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        loss_criterion = nn.MSELoss()

        n_iterations = math.ceil(n_samples / batch_size)
        epoch_loss = 0
        print(f"Step i/{n_iterations} [", end=" ")
        for i, (inputs, outputs, labels) in enumerate(dataloader):
            print(f"{i + 1},", end=" ")
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Feed
            outputs2 = self.feed(inputs)
            # Calculate loss
            loss = loss_criterion(outputs2, labels)
            # loss = loss_criterion(outputs.requires_grad_(True), labels)
            # Back propagate the loss
            loss.backward()
            # loss.backward(retain_graph=True)
            # Adjust the weights
            optimizer.step()
            # Save loss of this iteration to calculate mean loss at the end
            epoch_loss += loss
        print(f"]\nEpoch loss: {epoch_loss / n_iterations}")


class ReplayDataset(Dataset):
    def __init__(self, input, output, target):
        self.x = input
        self.output = output
        self.y = target
        self.n_samples = input.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.output[index], self.y[index]

    def __len__(self):
        return self.n_samples


def get_state(state, row_count, col_count):
    seed_pos = state.seed_pos
    one_pos = state.one_pos
    return [seed_pos[0] / row_count, seed_pos[1] / col_count, one_pos[0] / row_count, one_pos[1] / col_count]


if __name__ == '__main__':
    # Neural network
    model = ANN()
    model_target = ANN()
    # Instance of the game
    game_instance = env.Pygame()
    row_count = float(game_instance.row_count) - 1
    col_count = float(game_instance.col_count) - 1
    # Collecting data
    for epoch in range(max_epochs):
        print(f"Epoch: {epoch}")
        # Linearly decrease epsilon
        epsilon = 1.0 - epoch * (1.0 / max_epochs)
        # Containers to store the dataset (replay samples)
        batch_i = torch.Tensor(n_samples, 1, 4)
        batch_o = torch.Tensor(n_samples, 1, 4)
        batch_t = torch.Tensor(n_samples, 1, 4)
        # Current state is normalized a tuple(seed.x, seed.y, p_one.x, p_one.y)
        state_t = get_state(game_instance.game.reset(), row_count, col_count)
        with torch.no_grad():
            for i in range(n_samples):
                # Set tensor values
                input = torch.tensor([state_t])
                # Feed to check approximated Q-values
                output = model.feed(input)
                # Pick action (1 = up, 2 = right, 3 = down, 4 = left)
                action = None
                if np.random.random() > epsilon:
                    action = torch.argmax(output[0])    # greedy
                else:
                    action = np.random.randint(0, 4)    # random
                # Get state s_t + 1
                new_state, reward = game_instance.game.step(int(action) + 1, 1)
                state_t1 = get_state(new_state, row_count, col_count)
                # Decrease reward in case it moved towards wall
                if state_t == state_t1:
                    reward += -0.9
                # Get input given new state
                input1 = torch.tensor([state_t1])
                # Feed new state to get the max Q-value at new position
                output1 = model_target.feed(input1)
                # Calculate max future q
                max_future_q = float(torch.max(output1[0]))
                # Calculate new Q-value
                new_q_value = reward + (discount * max_future_q)
                target = output.detach().clone()
                # The values in output given input should be exactly the same except that
                # the Q-value with respect to action should be new_q_value
                target[0][action] = new_q_value
                # Assume new state
                state_t = state_t1
                # Save samples
                batch_i[i] = input.detach().clone()
                batch_o[i] = output.clone()
                batch_t[i] = target.detach().clone()
                # Draw game
                game_instance.pump()
                game_instance.render()
                time.sleep(0.000001)
        # Create data set
        print(f"Creating dataset composed of {n_samples} samples, with batch size of {batch_size} ")
        dataset = ReplayDataset(batch_i, batch_o, batch_t)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        # Train based on the replay
        model.backward(dataloader)
        # Update target network every 25 epochs
        if (epoch + 1) % 25 == 0:
            print("Updating target network")
            model_target = copy.deepcopy(model)

    # Testing phase
    state_t = get_state(game_instance.game.reset(), row_count, col_count)
    for t in range(test_iterations):
        if t % 100 == 0:
            print(f"Testing Iteration {t}")
        with torch.no_grad():
            # Set tensor values
            input = torch.tensor([state_t])
            # Feed to check approximated Q-values
            output = model.feed(input)
        print("input", input)
        print("output", output)

        # Define action
        action = torch.argmax(output[0])
        print(action)

        # Get state s_t + 1
        new_state, reward = game_instance.game.step(int(action) + 1, 1)
        state_t1 = get_state(new_state, row_count, col_count)

        state_t = state_t1
        # Pump events
        game_instance.pump()
        game_instance.render()
        time.sleep(0.5)