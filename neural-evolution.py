import multiprocessing
import time
import random
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal
from joblib import Parallel, delayed

import env.gridseedpygame as env
import env.game as gm

torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = 3
        self.outputs = 4
        self.normal = normal.Normal(0.0, 1.0)
        self.l1 = nn.Linear(self.inputs, 4)
        self.l2 = nn.Linear(4, 4)
        self.l3 = nn.Linear(4, self.outputs)

    def forward(self, x):
        with torch.no_grad():
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            x = F.softmax(self.l3(x), dim=1)
            return x

    def feed(self, x):
        outputs = self.forward(x)
        return outputs

    def get_state(self, state, row_count, col_count):
        player = state.one_pos
        seed = state.seed_pos
        adj = seed[0] - player[0]
        opp = seed[1] - player[1]
        hyp = math.sqrt(pow(adj, 2) + pow(opp, 2))
        sinx = adj / hyp
        cosx = opp / hyp
        max_hyp = math.sqrt(math.pow(row_count-1, 2) + math.pow(col_count-1, 2))
        hyp = hyp / max_hyp
        return [hyp, sinx, cosx]

    def disable_grad(self):
        for param in self.parameters():
            param.requires_grad = False

    def mutate(self, rate):
        for name, param in self.named_parameters():
            if len(param.shape) == 2:
                for i in range(param.shape[0]):
                    for j in range(param.shape[1]):
                        if random.random() <= rate:
                            param.data[i][j] = self.normal.sample()

            if len(param.shape) == 1:
                for i in range(param.shape[0]):
                    if random.random() <= rate:
                        param.data[i] = self.normal.sample()

class GeneticAlgorithm:
    def __init__(self):
        self.number_individuals = multiprocessing.cpu_count()
        self.iterations = 500
        self.pop = []
        self.pop_t1 = []
        self.fitness = torch.Tensor(self.number_individuals)
        self.fitness_t1 = torch.Tensor(self.number_individuals)
        self.init_population()

    def init_population(self):
        for i in range(self.number_individuals):
            self.pop.append(ANN())
            self.pop_t1.append(ANN())
            self.pop[i].disable_grad()
            self.pop_t1[i].disable_grad()
            self.fitness[i] = -100
            self.fitness_t1[i] = -100

    # Evaluate only with game logic
    def run_game(self, id):
        fitness = 0
        game_instance = gm.Game()
        row_count = float(gm.ROW_COUNT)
        col_count = float(gm.COLUMN_COUNT)
        game_state = self.pop_t1[id].get_state(game_instance.reset(), row_count, col_count)
        for i in range(250):
            input = torch.tensor([game_state])
            output = self.pop_t1[id].feed(input)
            action = torch.argmax(output[0])
            new_state, reward = game_instance.step(int(action) + 1)
            if reward is None:
                fitness = -1
                break
            fitness += reward
            game_state = self.pop_t1[id].get_state(new_state, row_count, col_count)

        return fitness

    def run_game_graphic(self, id):
        index = torch.argmax(self.fitness)
        fitness = 0
        game_instance = env.Pygame()
        row_count = float(game_instance.row_count)
        col_count = float(game_instance.col_count)
        game_state = self.pop_t1[id].get_state(game_instance.game.reset(), row_count, col_count)
        for i in range(50):
            input = torch.tensor([game_state])
            output = self.pop_t1[id].feed(input)
            action = torch.argmax(output[0])
            new_state, reward = game_instance.game.step(int(action) + 1)
            if reward is None:
                break
            fitness += reward
            game_state = self.pop_t1[id].get_state(new_state, row_count, col_count)
            if id == index.item():
                game_instance.pump()
                game_instance.render()
                time.sleep(0.05)
        game_instance.quit_game()
        del game_instance
        return fitness

    def evaluate(self):
        fitness = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(self.run_game)(i) for i in range(self.number_individuals))
        return fitness

    def select_two(self):
        index1 = self.tournament_selection(2)
        index2 = self.tournament_selection(2)
        while index1 == index2:
            index2 = self.tournament_selection(2)
        return index1, index2

    def tournament_selection(self, tournament_size):
        index1 = np.random.randint(0, self.number_individuals)
        index2 = np.random.randint(0, self.number_individuals)
        while index2 == index1:
            index2 = np.random.randint(0, self.number_individuals)
        if self.fitness[index1] > self.fitness[index2]:
            return index1
        if self.fitness[index2] > self.fitness[index1]:
            return index2
        else:
            if np.random.rand() < 0.5:
                return index1
            else:
                return index2

    def crossover(self, par1, par2, child):
        for name, param in self.pop_t1[child].named_parameters():
            if len(param.shape) == 2:
                for i in range(param.shape[0]):
                    for j in range(param.shape[1]):
                        if np.random.rand() <= 0.5:
                            param.data[i][j] = self.pop[par1].state_dict()[name].data[i][j]
                        else:
                            param.data[i][j] = self.pop[par2].state_dict()[name].data[i][j]
            if len(param.shape) == 1:
                for i in range(param.shape[0]):
                    if np.random.rand() <= 0.5:
                        param.data[i] = self.pop[par1].state_dict()[name].data[i]
                    else:
                        param.data[i] = self.pop[par2].state_dict()[name].data[i]

    def optimize(self):
        with torch.no_grad():
            for iteration in range(self.iterations):
                mutation_rate = 0.01 + (0.09 - iteration * (0.09 / self.iterations))
                if iteration % 10 == 0:
                    print(f"Iteration {iteration}, mutation: {mutation_rate}")
                    print('fitness  ', self.fitness)
                    print('fitnesst1       ', self.fitness_t1)
                for i in range(self.number_individuals):
                    index1, index2 = self.select_two()
                    self.crossover(index1, index2, i)
                    self.pop_t1[i].mutate(mutation_rate)
                self.fitness_t1 = self.evaluate()
                for i in range(self.number_individuals):
                    if self.fitness[i] <= self.fitness_t1[i]:
                        self.fitness[i] = self.fitness_t1[i]
                        self.pop[i].load_state_dict(copy.deepcopy(self.pop_t1[i].state_dict()))

        best = self.get_best()
        torch.save(self.pop[best].state_dict(), f'neural-evo-models/model_score{int(self.fitness[best])}')

    def get_best(self):
        return torch.argmax(self.fitness)

    def test_model(self, model_name):
        game_instance = env.Pygame()
        model = ANN()
        model.load_state_dict(torch.load('neural-evo-models/' + model_name))
        for param in model.parameters():
            print(param)
        row_count = float(game_instance.row_count)
        col_count = float(game_instance.col_count)
        state_t = model.get_state(game_instance.game.reset(), row_count, col_count)
        for t in range(1000):
            input = torch.tensor([state_t])
            output = model.feed(input)
            action = torch.argmax(output[0])
            new_state, reward = game_instance.game.step(int(action) + 1)
            state_t1 = model.get_state(new_state, row_count, col_count)
            if reward is None:
                print('Dead by wall')
                break
            state_t = state_t1
            # Render
            game_instance.pump()
            game_instance.render()
            time.sleep(0.01)


def main():
    with torch.no_grad():
        ga = GeneticAlgorithm()
        ga.optimize()
        ga.test_model('model_score73')


if __name__ == '__main__':
    # with torch.no_grad():
    #     print(model1.state_dict()['l1.weight'])
    #     print(model2.state_dict()['l1.weight'])
    #
    #     1st
    #     state_dic1 = model1.state_dict()
    #     state_dic2 = model2.state_dict()
    #     state_dic1['l1.weight'] = state_dic2['l1.weight']
    #     model1.load_state_dict(state_dic1)
    #     print(model1.state_dict()['l1.weight'])
    #     END Strategy one
    #
    #     2nd
    #     model1.l1.weight.data = torch.clone(model2.l1.weight.data)
    #     model2.l1.weight.data[0][0] = 0
    #     print(model1.state_dict()['l1.weight'])
    #     print(model2.state_dict()['l1.weight'])

    # Run game to test
    # game_instance = env.Pygame()
    # game_instance.run()
    # quit()
    main()

