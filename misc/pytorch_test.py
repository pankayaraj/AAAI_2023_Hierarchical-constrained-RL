import torch
from torch import nn
import numpy as np
from envs.past.grid.safety_gridworld import PitWorld
from models.ours.grid_model import OneHotCostAllocator

env = PitWorld(size = 14,
                max_step = 200,
                per_step_penalty = -1.0,
                goal_reward = 1000.0,
                obstace_density = 0.3,
                constraint_cost = 10.0,
                random_action_prob = 0.005,
                one_hot_features=True,
                rand_goal=True,)
state = env.reset()
print(np.concatenate((state, state)).shape)


cost = np.array([10.0, 10.0, 10.0]).transpose()
state = np.array([state, state, state])
t_state = torch.Tensor(state)

from models.ours.grid_model import OneHotCostAllocator, OneHotValueNetwork

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
s = env.reset()

shape = (s.shape[0]+1, )

C = OneHotCostAllocator(num_inputs=shape).to(device=device)
V = OneHotValueNetwork(num_inputs=shape).to(device=device)

#print(C(state, cost))
#print(V(t_state))

s = torch.FloatTensor(s).to(device)
A = torch.FloatTensor([100]).to(device=device)
X = torch.cat((s,A))
w = C(X)
print(w)
B = w[0]*A
C = w[1]*A

print(B.backward())
print(A, B, C)