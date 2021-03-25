import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal

import env.gridseedpygame as env
from neuro_optimizer.genetic_algorithm import GeneticAlgorithm as GA
from neuro_optimizer.differential_evolution import DifferentialEvolution as DE

torch.set_grad_enabled(False)
torch.autograd.set_grad_enabled(False)


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = 3
        self.outputs = 4
        self.normal = normal.Normal(0.0, 1.0)
        self.l1 = nn.Linear(self.inputs, 8)
        self.l2 = nn.Linear(8, 8)
        self.l3 = nn.Linear(8, self.outputs)

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

    def init_xavier(self):
        for name, param in self.named_parameters():
            # Init weights with xavier
            if len(param.shape) == 2:
                nn.init.xavier_uniform_(param)
            # Set bias to zero
            if len(param.shape) == 1:
                param.data.fill_(0.0)

        # with torch.no_grad():
        #     # Xavier initialization
        #     nn.init.xavier_uniform_(self.l1.weight)
        #     # Set bias to zero
        #     self.l1.bias.data.fill_(0.0)

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


def main():
    # ga = GA(ANN)
    # model_name, best_fitness = ga.optimize()
    # ga.test_model(model_name)

    de = DE(ANN)
    model_name, best_fitness = de.optimize()
    de.test_model(model_name)
    quit()


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

