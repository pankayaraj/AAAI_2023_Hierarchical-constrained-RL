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
from agents.ours.hrl_sarsa_agent_dummy import Dummy
from agents.ours.hrl_safe_dummy_2 import Dummy_2
from agents.ours.safe_lower_hrl_sarsa_agent import SAFE_LOWER_HRL_Discrete_Goal_SarsaAgent
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
    #agent = SAFE_LOWER_HRL_Discrete_Goal_SarsaAgent(args, env, save_dir=args.save_dir, exp_no=args.exp_no)
    #agent = Dummy(args, env, save_dir=args.save_dir, exp_no=args.exp_no)
    agent =  Dummy_2(args, env, save_dir=args.save_dir, exp_no=args.exp_no)
else:
    raise Exception("Not implemented yet")


print("experiment_numer " + str(args.exp_no))

agent.save_dir = "Temporary/"
# start the run process here
agent.load()

import time
next_state = None

avg_reward = []
avg_constraint  = []

print(agent.env.to_string())
print(agent.eval_env.to_string())
for i in range(5):
    state = agent.eval_env.reset()
    previous_state = torch.FloatTensor(state)
    done = False
    ep_reward = 0
    ep_constraint = 0
    ep_len = 0
    start_time = time.time()

    IR = []
    Goals = []
    CS = []
    lower_cost_constraint = torch.zeros(agent.args.num_envs).float().to(agent.device)
    lower_cost_constraint[0] = 10  # manually set cost division for now
    current_cost = torch.zeros(agent.args.num_envs, 1).float().to(agent.device)

    while not done:

                # convert the state to tensor
        state = torch.FloatTensor(state).to(agent.device)

                # get the goal
        goal = agent.pi_meta(state=state, greedy_eval=greedy_eval)

        x_g, y_g = agent.G.convert_value_to_coordinates(agent.G.goal_space[goal.item()])
        Goals.append((x_g, y_g))

        goal = torch.LongTensor(goal).unsqueeze(1).to(agent.device)

        goal_hot_vec = agent.G.covert_value_to_hot_vec(goal)
        goal_hot_vec = torch.FloatTensor(goal_hot_vec).to(agent.device)

        t_lower = 0
        ir = 0
        while t_lower <= agent.args.max_ep_len_l-1:

            action = agent.safe_deterministic_pi_lower(state=state, goal=goal_hot_vec, d_low=lower_cost_constraint.detach(), current_cost=current_cost, greedy_eval=greedy_eval)
                        # print(torch.equal(state, previous_state), self.G.convert_hot_vec_to_value(state), self.G.convert_hot_vec_to_value(goal_hot_vec))
                        # print(self.dqn_lower(torch.cat((state, goal_hot_vec))), t_lower)

            next_state, reward, done, info = agent.eval_env.step(action.item())
            ep_reward += reward
            ep_len += 1
            ep_constraint += info[agent.cost_indicator]


            next_state = torch.FloatTensor(next_state).to(agent.device)

                        # update the state
            previous_state = state
            state = next_state

            current_cost = torch.FloatTensor([info[agent.cost_indicator] * (1.0 - done)]).unsqueeze(1).to(agent.device)

            instrinc_reward = agent.G.intrisic_reward(current_state=next_state,
                                                                 goal_state=goal_hot_vec)
            ir += instrinc_reward
            t_lower += 1

            done_l = agent.G.validate(current_state=next_state, goal_state=goal_hot_vec)
            if done_l or done:
                break

        IR.append(ir)
        x_c, y_c = agent.G.convert_value_to_coordinates(agent.G.convert_hot_vec_to_value(next_state).item())
        CS.append((x_c, y_c))

    avg_reward.append(ep_reward)
    avg_constraint.append(ep_constraint)

print(avg_reward, avg_constraint, Goals, CS, Goals)