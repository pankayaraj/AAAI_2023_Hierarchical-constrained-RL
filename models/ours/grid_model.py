import math
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class OneHotDQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(OneHotDQN, self).__init__()

        num_feats = num_inputs[0]
        self.dummy_param = nn.Parameter(torch.empty(0))  # to get the device name designated to the module

        self.q_layer = nn.Sequential(
            nn.Linear(num_feats, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, state):
        # to get the device assigned to the module at initalization
        device = self.dummy_param.device
        if type(state) != torch.Tensor:
            state = torch.Tensor(state).to(device=device)

        q_vals = self.q_layer(state)
        return q_vals

class OneHotValueNetwork(nn.Module):
    def __init__(self, num_inputs):
        super(OneHotValueNetwork, self).__init__()

        num_feats = num_inputs[0]
        self.dummy_param = nn.Parameter(torch.empty(0))  # to get the device name designated to the module
        self.value = nn.Sequential(
            nn.Linear(num_feats, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        # to get the device assigned to the module at initalization
        device = self.dummy_param.device
        if type(state) != torch.Tensor:
            state = torch.Tensor(state).to(device=device)

        val = self.value(state)
        return val

class OneHotCostAllocator(nn.Module):

    def __init__(self, num_inputs):
        super(OneHotCostAllocator, self).__init__()

        input_dim = num_inputs[0]+1 #+1 is for cost as an input
        self.dummy_param = nn.Parameter(torch.empty(0)) #to get the device name designated to the module

        self.cost_allocator = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, state, cost):

        #to get the device assigned to the module at initalization
        device = self.dummy_param.device

        if type(cost) != torch.Tensor:
            cost = torch.reshape(torch.Tensor(np.array(cost)).to(device), shape=(-1,1))
        if type(state) != torch.Tensor:
            state = torch.Tensor(state).to(device=device)

        inp = torch.cat((state, cost), dim=1)
        cost_weights = self.cost_allocator(inp)

        return cost_weights