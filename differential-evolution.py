import env.gridseedpygame as env
import numpy as np
import time
import copy
import multiprocessing
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ANN(nn.Module):
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
    def forward(self, x):
        # Passes x through layer one and activate with rectified linear unit function
        with torch.no_grad():
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            x = F.relu(self.l3(x))
            x = F.softmax(self.l4(x), dim=1)
            return x

    def feed(self, x):
        outputs = self.forward(x)
        return outputs

    def flatten_params(self):
        l = [torch.flatten(p) for p in self.parameters()]
        indices = []
        s = 0
        for p in l:
            size = p.shape[0]
            indices.append((s, s+size))
            s += size
        flat = torch.cat(l).view(-1, 1)
        return {"params": flat, "indices": indices}

    def recover_flattened(self, flat_params, indices):
        l = [flat_params[s:e] for (s, e) in indices]
        for i, p in enumerate(self.parameters()):
            l[i] = l[i].view(*p.shape)
            p.data = l[i]

    # Normalize the game state
    def get_state(self, state, row_count, col_count):
        seed_pos = state.seed_pos
        one_pos = state.one_pos
        return [seed_pos[0] / row_count, seed_pos[1] / col_count, one_pos[0] / row_count, one_pos[1] / col_count]


class GeneticAlgorithm:
    def __init__(self, dimension):
        # Dimension of the representation
        self.dimension = dimension
        # GA's parameter
        self.number_individuals = 16
        self.iterations = 30
        self.crossover_rate = 0.9
        self.mutation_rate = 0.1
        # Population t
        self.pop = []
        # Fitness of self.pop
        self.fitness = []
        # Population t+1
        self.pop_t1 = []
        # Fitness of self.pop_t1
        self.fitness_t1 = []
        # ANN for each individual in pop
        self.anns = []
        # Initialize population and ANNs
        self.init_population()

    def init_population(self):
        for i in range(self.number_individuals):
            self.anns.append(ANN())
            self.pop.append(self.anns[i].flatten_params())
        self.fitness = self.evaluate()

    def run_game(self, id):
        # Fitness of the id-th ANN
        fitness = 0

        # Instance of the game used to evaluate id-th ANN
        game_instance = env.Pygame()
        row_count = float(game_instance.row_count) - 1
        col_count = float(game_instance.col_count) - 1

        # Current state of the game
        game_state = self.anns[id].get_state(game_instance.game.reset(), row_count, col_count)
        for i in range(10):
            # Input based on the state
            input = torch.tensor([game_state])
            # Feed the input
            output = self.anns[id].feed(input)
            # Check action with highest probability
            action = torch.argmax(output[0])
            # Get state s_t + 1
            new_state, reward = game_instance.game.step(int(action) + 1, 1)
            fitness += max(0, reward)
            game_state = self.anns[id].get_state(new_state, row_count, col_count)
            # game_instance.pump()
            # game_instance.render()
            # time.sleep(0.01)
        # Quit the game
        game_instance.quit_game()
        del game_instance

        return fitness

    def evaluate(self):
        # Run self.number_individuals games in parallel
        fitness = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(self.run_game)(i) for i in range(self.number_individuals))
        return fitness

    # Tournament selection
    def tournament_selection(self, tournament_size):
        # Pick two individuals at random
        index1 = np.random.randint(0, self.number_individuals)
        index2 = np.random.randint(0, self.number_individuals)
        while index2 == index1:
            index2 = np.random.randint(0, self.number_individuals)
        # The one with highest fitness win the tournament
        if self.fitness[index1] > self.fitness[index2]:
            return index1
        if self.fitness[index2] > self.fitness[index1]:
            return index2
        else:
            if np.random.rand() < 0.5:
                return index1
            else:
                return index2

    # Uniform crossover
    def crossover(self, parent1, parent2):
        offspring1 = torch.zeros(self.dimension, 1)
        offspring2 = torch.zeros(self.dimension, 1)

        for i in range(self.dimension):
            if np.random.rand() <= 0.5:
                offspring1[i][0] = self.pop[parent1]['params'][i][0]
                offspring2[i][0] = self.pop[parent2]['params'][i][0]
            else:
                offspring1[i][0] = self.pop[parent2]['params'][i][0]
                offspring2[i][0] = self.pop[parent1]['params'][i][0]

        indices1 = self.pop[parent1]['indices']
        indices2 = self.pop[parent2]['indices']
        return {"params": offspring1, "indices": indices1}, {"params": offspring2, "indices": indices2}

    # Uniform mutation
    def mutate(self, individual1, individual2):
        if np.random.rand() <= self.mutation_rate:
            index = np.random.randint(self.dimension)
            individual1['params'][index][0] += 2 * (np.random.rand() - 0.5)

        if np.random.rand() <= self.mutation_rate:
            index = np.random.randint(self.dimension)
            individual2['params'][index][0] += 2 * (np.random.rand() - 0.5)

    def optimize(self):
        for iteration in range(self.iterations):
            print(f"Iteration {iteration}")
            i = 0
            self.pop_t1 = []
            while i < self.number_individuals:
                # Pick two distinct individuals as parents
                parent1 = self.tournament_selection(2)
                parent2 = self.tournament_selection(2)
                while parent1 == parent2:
                    parent2 = self.tournament_selection(2)
                # Perform crossover
                off1, off2 = self.crossover(parent1, parent2)
                # Perform mutation
                self.mutate(off1, off2)
                # Add to the temporary population
                self.pop_t1.append(off1)
                self.pop_t1.append(off2)
                # Update corresponding ANN parameters
                self.anns[i].recover_flattened(off1['params'], off1['indices'])
                self.anns[i + 1].recover_flattened(off2['params'], off2['indices'])
                i += 2

            # Evaluate new individuals
            self.fitness_t1 = self.evaluate()
            for i in range(self.number_individuals):
                if self.fitness[i] <= self.fitness_t1[i]:
                    self.fitness[i] = self.fitness_t1[i]
                    self.pop[i] = self.pop_t1[i]

            print(self.fitness)

    def get_best(self):
        index = np.argmax(self.fitness)
        return self.pop[index]

if __name__ == '__main__':
    # model = ANN()
    # params = model.flatten_params()
    # params['params'][0] = 0.0
    # params['params'][1] = 0.0
    # model.recover_flattened(params['params'], params['indices'])

    numb_param = len(ANN().flatten_params()['params'])
    ga = GeneticAlgorithm(numb_param)
    ga.optimize()

    # Instance of the game
    game_instance = env.Pygame()
    # Model
    model = ANN()
    best = ga.get_best()
    model.recover_flattened(best['params'], best['indices'])

    for param in model.parameters():
        print(param)

    row_count = float(game_instance.row_count) - 1
    col_count = float(game_instance.col_count) - 1
    state_t = model.get_state(game_instance.game.reset(), row_count, col_count)
    for t in range(100):
        # Set tensor values
        input = torch.tensor([state_t])
        # Feed
        output = model.feed(input)
        # Define action
        action = torch.argmax(output[0])

        # Get state s_t + 1
        new_state, reward = game_instance.game.step(int(action) + 1, 1)
        state_t1 = model.get_state(new_state, row_count, col_count)

        state_t = state_t1
        # Pump events
        game_instance.pump()
        game_instance.render()
        time.sleep(0.1)

