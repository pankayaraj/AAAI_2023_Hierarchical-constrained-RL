import os
import numpy as np
import random
import torch
import shutil

from common.past.utils import *
#from common.past.arguments import get_args
from common.ours.arguments import get_args

# SARSA agents
from agents.past.sarsa_agent import SarsaAgent
from agents.past.safe_sarsa_agent import SafeSarsaAgent
from agents.past.lyp_sarsa_agent import LypSarsaAgent
from agents.ours.hrl_sarsa_agent import HRL_Discrete_Goal_SarsaAgent

from agents.ours.hrl_safe_lower_with_manual_constraint import HRL_Discrete_Safe_Lower_Manual_Constraints
from agents.ours.hrl_safe_with_global_costraints import HRL_Discrete_Safe_Global_Constraint
from agents.ours.hrl_safe_with_global_constraints_dual import HRL_Discrete_Safe_Global_Constraint_Dual
from agents.ours.hrl_safe_with_cost_allocation import HRL_Discrete_Safe_Lower_Cost_Alloc
from agents.ours.hrl_sarsa_cost_allocation_only import HRL_Discrete_Safe_Lower_Cost_Allocation_Only
from agents.ours.hrl_safe_upper_bvf_only_lower_lagarangian import HRL_Discrete_Safe_Upper_BVF_Only_lower_Lagrangian

"""
# A2C based agents
from agents.a2c_agent import A2CAgent
from agents.lyp_a2c_agent import LyapunovA2CAgent
from agents.safe_a2c_v2_agent import SafeA2CProjectionAgent

# PPO based agents
from agents.ppo import PPOAgent
from agents.safe_ppo import SafePPOAgent
from agents.lyp_ppo import LyapunovPPOAgent
# target based agents
from agents.target_agents.target_bvf_ppo import TargetBVFPPOAgent
from agents.target_agents.target_lyp_ppo import TargetLypPPOAgent
"""


# get the args from argparse
args = get_args()
# dump the args
log(args)


# initialize a random seed for the experiment
seed = np.random.randint(1,1000)
args.seed = seed
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# pytorch multiprocessing flag
torch.set_num_threads(1)

# check the device here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


agent = None

# create the env here
env = create_env(args)

"""
# create the agent here
# PPO based agents
if args.agent == "ppo":
    agent = PPOAgent(args, env)
elif args.agent == "bvf-ppo":
    if args.target:
        agent = TargetBVFPPOAgent(args, env)
    else:
        agent = SafePPOAgent(args, env, writer=tb_writer)
elif args.agent == "lyp-ppo":
    if args.target:
        agent = TargetLypPPOAgent(args, env)
    else:
        agent = LyapunovPPOAgent(args, env)
# A2C based agents
elif args.agent == "a2c":
    agent = A2CAgent(args, env, writer=tb_writer)
elif args.agent == "safe-a2c":
    agent = SafeA2CProjectionAgent(args, env, writer=tb_writer)
elif args.agent == "lyp-a2c":
    agent = LyapunovA2CAgent(args, env, writer=tb_writer)
"""

from multiprocessing import freeze_support, set_start_method
args.exp_no = "2"
greedy_eval = True
# don't use tb on cluster
tb_writer = None

freeze_support()
#  SARSA based agent
if args.agent == "sarsa":
    agent = SarsaAgent(args, env, writer=tb_writer, save_dir=args.save_dir, exp_no=args.exp_no)
elif args.agent == "bvf-sarsa":
    agent = SafeSarsaAgent(args, env, writer=tb_writer, save_dir=args.save_dir, exp_no=args.exp_no)
elif args.agent == "lyp-sarsa":
    agent = LypSarsaAgent(args, env, writer=tb_writer)
elif args.agent == "hrl-sarsa":
    agent = HRL_Discrete_Goal_SarsaAgent(args, env, save_dir=args.save_dir, exp_no=args.exp_no)
elif args.agent == "safe-lower-hrl-sarsa":
    agent =  HRL_Discrete_Safe_Lower_Manual_Constraints(args, env, save_dir=args.save_dir, exp_no=args.exp_no)
elif args.agent == "safe-global-hrl-sarsa":
    agent = HRL_Discrete_Safe_Global_Constraint(args, env, save_dir=args.save_dir, exp_no=args.exp_no)
elif args.agent == "safe-dual-global-hrl-sarsa":
    agent = HRL_Discrete_Safe_Global_Constraint_Dual(args, env, save_dir=args.save_dir, exp_no=args.exp_no)
elif args.agent == "safe-cost-alloc-lower-hrl-sarsa":
    agent = HRL_Discrete_Safe_Lower_Cost_Alloc(args, env, save_dir=args.save_dir, exp_no=args.exp_no)
elif args.agent == "safe-cost-alloc-lower-only-hrl-sarsa":
    agent = HRL_Discrete_Safe_Lower_Cost_Allocation_Only(args, env, save_dir=args.save_dir, exp_no=args.exp_no)
elif args.agent == "safe-upper_bvf_lower_lagrangian":
    agent = HRL_Discrete_Safe_Upper_BVF_Only_lower_Lagrangian(args, env, save_dir=args.save_dir, exp_no=args.exp_no)
else:
    raise Exception("Not implemented yet")


print("experiment_numer " + str(args.exp_no))

agent.save_dir = ""
# start the run process here
import time


#eps_of_interest = range(500)


#eps_of_interest = [408, 261, 425, 451]
eps_of_interest = [25, 151]
end_state = []
for i in eps_of_interest:

    print("=================================")
    print(i)
    agent.load()
    next_state = None


    avg_reward = []
    avg_constraint  = []

    t_lower = 0
    ir = 0
    C = 0
    R = 0

    state = agent.eval_env.reset()
    state = torch.FloatTensor(state).to(device=agent.device)
    goal  = torch.LongTensor([1])
    goal_hot_vec = agent.G.covert_value_to_hot_vec(goal)
    goal_hot_vec = torch.FloatTensor(goal_hot_vec).to(agent.device)

    #print(agent.safe_deterministic_pi_upper(state, greedy_eval=True))
    initial_state = state
    S_L = []
    I_R = []

    current_cost = torch.zeros(agent.args.num_envs, 1).float().to(agent.device)

    while t_lower <= agent.args.max_ep_len_l - 1:


        # action = agent.safe_deterministic_pi_lower(state=state, goal=goal_hot_vec, goal_discrete=goal,  current_cost=current_cost, greedy_eval=greedy_eval)
        action = agent.safe_deterministic_pi_lower(state=state, initial_state=initial_state, current_cost=current_cost, goal=goal_hot_vec, goal_discrete=goal,
                                                    greedy_eval=True)

        next_state, reward, done, info = agent.eval_env.step(action.item())

        next_state = torch.FloatTensor(next_state).to(agent.device)

        # update the state
        previous_state = state
        state = next_state

        current_cost = torch.FloatTensor([info[agent.cost_indicator] * (1.0 - done)]).unsqueeze(1).to(agent.device)

        instrinc_reward = agent.G.intrisic_reward(current_state=next_state,
                                                  goal_state=goal_hot_vec)

        x_c, y_c = agent.G.convert_value_to_coordinates(agent.G.convert_hot_vec_to_value(next_state).item())
        S_L.append((x_c, y_c))
        I_R.append(instrinc_reward)

        C += info[agent.cost_indicator]
        R += instrinc_reward

        ir += instrinc_reward
        t_lower += 1

        done_l = agent.G.validate(current_state=next_state, goal_state=goal_hot_vec)
        if done_l or done:
            break

    end_state.append(next_state)
    print(C, "cost")
    # print(agent.cost_lower_model(psg), C, goal)
    # print(agent.dqn_lower(psg), R)


    print(S_L)
    print(I_R)


import math
from math import sqrt
import matplotlib.pyplot as plt

def convert_int_to_coordinates(value, grid_size):
    x = (value)%grid_size
    y = (value)//grid_size
    return x, y

def convert_coordinated_into_int(x, y, grid_size):
    value = y*grid_size + x
    return value

def convert_value_to_hot_vec(value, grid_size):

    hot_vector = [0 for _ in range(grid_size*grid_size)]
    hot_vector[value] = 1
    return np.array(hot_vector).flatten()

def euclidian_distance(x1, y1, x2, y2, grid_size):

    max = (math.sqrt(2*grid_size**2))

    d = (x1-x2)**2 + (y1-y2)**2
    return sqrt(d)

grid_size = agent.G.grid_size
distance = []

x1 = 16
y1 = 8

for i in range(len(end_state)):
    x2, y2 = agent.G.convert_value_to_coordinates(agent.G.convert_hot_vec_to_value(end_state[i]).item())
    distance.append(-euclidian_distance(x1, y1, x2, y2, grid_size))

plt.plot(distance)
#plt.show()