import torch

from common.ours.grid.utils import Goal_Space
import numpy as np
from PIL import Image

G = Goal_Space([], 18)


A  = []
for i in range(18): #y
    a = []
    for j in range(18): #x
        a.append(G.convert_cooridnates_to_value(j, i))
    A.append(a)


start = 180

goal = [212, 86, 135, 282, 147, 200]
print(G.convert_cooridnates_to_value(16, 10))
print(G.convert_value_to_coordinates(298))
A = np.array(A)
print(A)


from common.past.utils import create_env
from common.past.arguments import get_args

args = get_args()
args.env_name = "grid_key"
env = create_env(args)

print(env.to_string())

state = env.reset()
state = torch.FloatTensor(state)
#print(state.shape)
#print(env.start_x, env.start_y, env.goal_x, env.goal_y)
#print(G.convert_hot_vec_to_value(state).item())
my = env.size-2
mx = env.size-2

print(G.convert_cooridnates_to_value(1, my//2+1))
print(G.convert_cooridnates_to_value(mx, my//2))


