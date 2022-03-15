import math

import torch
import numpy as np
from math import sqrt


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

def euclidian_distance(value1, value2, grid_size):
    x1, y1 = convert_int_to_coordinates(value1, grid_size)
    x2, y2 = convert_int_to_coordinates(value2, grid_size)
    max = (math.sqrt(2*grid_size**2))

    d = (x1-x2)**2 + (y1-y2)**2
    return sqrt(d)/max


#class that defines goal space
class Goal_Space():
    def __init__(self, goal_space, grid_size, intrinsic_reward_type="eculidian distance"):

        #goal_space: A list of indices which corresponds to the location on grid as indicated by one-hot vector
        self.grid_size = grid_size
        self.goal_space = goal_space
        self.intrinsic_reward_type = intrinsic_reward_type

        self.action_shape = (len(goal_space), 1)

    def find_shortest_goal_reward(self, current_state):
        current_value = torch.argmax(current_state).item()
        rewards = [-euclidian_distance(current_value, g, self.grid_size) for g in self.goal_space]

        return max(rewards)

    def validate_done(self, current_state):
        current_value = torch.argmax(current_state).item()
        if current_value in self.goal_space:
            return True
        else:
            return False

    def intrisic_reward(self, current_state, goal_state):
        if type(current_state) != torch.Tensor:
            current_state = torch.Tensor(current_state)
        if type(goal_state) != torch.Tensor:
            goal_state = torch.Tensor(goal_state)

        current_value = torch.argmax(current_state).item()
        goal_value = torch.argmax(goal_state).item()

        #print(current_value, goal_value)
        if self.intrinsic_reward_type == "eculidian distance":
            f = euclidian_distance(current_value, goal_value, self.grid_size)
            return -f
        else:
            raise Exception("Not Implemented")

    def action_to_goal(self, action):
        #get the corresponding goal for the discrete action
        return convert_value_to_hot_vec(self.goal_space[action], self.grid_size)

    def validate(self, current_state, goal_state):
        if type(current_state) != torch.Tensor:
            current_state = torch.Tensor(current_state)
        if type(goal_state) != torch.Tensor:
            goal_state = torch.Tensor(goal_state)



        return torch.equal(current_state, goal_state)

    def covert_value_to_hot_vec(self, goal):
        value = self.goal_space[goal]
        return convert_value_to_hot_vec(value, self.grid_size)

    def convert_value_to_coordinates(self, value):
        return convert_int_to_coordinates(value, self.grid_size)

    def convert_cooridnates_to_value(self, x, y):
        return convert_coordinated_into_int(x, y, self.grid_size)

    def convert_hot_vec_to_value(self, hot_vec):
        return torch.argmax(hot_vec)


class Cost_Space():

    def __init__(self, cost_space, cost_mapping):

        self.cost_space = cost_space
        self.cost_mapping = cost_mapping

        self.cost_shape = (len(cost_space), 1)

    def get_cost_weight(self, cost_index):
       
        return self.cost_mapping[cost_index]


"""
G = Goal_Space([1,10], 9)
print(convert_int_to_coordinates(34, 9))
print(convert_coordinated_into_int(6, 3, 9))
print(convert_value_to_hot_vec(35, 9))

S1 = convert_value_to_hot_vec(2,9)
S2 = convert_value_to_hot_vec(32,9)

print(G.intrisic_reward(S1, S2))
"""
import gym

from datetime import datetime
import sys

from collections import namedtuple

from rllab.misc import ext

from envs.ours.gird.safety_gridworld_with_key import PitWorld_Key

def create_env_hrl(args):
    """
    the main method which creates any environment
    """
    env = None

    if args.env_name == "pg":
        # create point gather envrionment
        # env = PointGatherEnv(n_apples=2,
        #                      n_bombs=8,
        #                      apple_reward=+10,
        #                      bomb_cost=10, #bomb inherently negative
        #                      max_ep_len = 15,
        #                      )
        env = create_PointGatherEnv()
    elif args.env_name == "pc":
        # create Point Circle
        env = SafePointEnv(circle_mode=True,
                           xlim=2.5,
                           abs_lim=True,
                           target_dist=15,
                           max_ep_len = 65,
                           )
    elif args.env_name == "cheetah":
        # create Point Circle as per Lyp PG
        env = SafeCheetahEnv(limit=1,
                             max_ep_len=200)
    elif args.env_name == "grid":
        # create the grid with pits env
        env = PitWorld_Key(size = 18,
                       max_step = 200,
                       per_step_penalty = -1.0,
                       goal_reward = 1000.0,
                       obstace_density = 0.3,
                       constraint_cost = 10.0,
                       random_action_prob = 0.005,
                       one_hot_features=True,
                       rand_goal=True, # for testing purposes
                       )
    else:
        raise Exception("Not implemented yet")

    return env
