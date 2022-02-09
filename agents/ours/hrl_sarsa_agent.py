import math
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import gym
import copy
import numpy as np


from common.past.utils import *
from common.ours.grid.utils import Goal_Space

from common.past.multiprocessing_envs import SubprocVecEnv
from torchvision.transforms import ToTensor

from models.past.grid_model import OneHotDQN

from common.past.schedules import LinearSchedule, ExponentialSchedule

class HRL_Discrete_Goal_SarsaAgent(object):

    def __init__(self,
                 args,
                 env,
                 goal_space,
                 writer = None):
        """
        init the agent here
        """

        self.grid_size = 14
        self.eval_env = copy.deepcopy(env)
        self.args = args

        self.G = Goal_Space(goal_space=goal_space,grid_size=self.grid_size)

        s = env.reset()
        self.state_dim = s.shape
        self.action_dim = env.action_space.n
        self.goal_dim = self.G.action_shape
        self.goal_state_dim = np.concatenate((s,s)).shape

        self.device = torch.device("cuda" if (torch.cuda.is_available() and  self.args.gpu) else "cpu")

        # set the same random seed in the main launcher
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        if self.args.gpu:
            torch.cuda.manual_seed(self.args.seed )

        self.writer = writer

        if self.args.env_name == "grid":
            self.dqn_meta = OneHotDQN(self.state_dim, self.goal_dim).to(self.device)
            self.dqn_meta_target = OneHotDQN(self.state_dim, self.goal_dim).to(self.device)

            self.dqn_lower = OneHotDQN(self.goal_state_dim, self.goal_dim).to(self.device)
            self.dqn_lower_target = OneHotDQN(self.goal_state_dim, self.goal_dim).to(self.device)
        else:
            raise Exception("not implemented yet!")

        # copy parameters
        self.dqn_meta_target.load_state_dict(self.dqn_meta.state_dict())
        self.dqn_lower_target.load_state_dict(self.dqn_lower.state_dict())

        self.optimizer_meta = torch.optim.Adam(self.dqn_meta.parameters(), lr=self.args.lr)
        self.optimizer_lower = torch.optim.Adam(self.dqn_lower.parameters(), lr=self.args.lr)

        # for actors
        def make_env():
            def _thunk():
                env = create_env(args)
                return env

            return _thunk

        envs = [make_env() for i in range(self.args.num_envs)]
        self.envs = SubprocVecEnv(envs)

        self.total_steps = 0
        self.total_lower_time_steps = 0
        self.total_meta_time_steps = 0
        self.num_episodes = 0

        #different epsilon for different levels
        self.eps_u_decay = LinearSchedule(50000 * 200, 0.01, 1.0)
        self.eps_l_decay = LinearSchedule(50000 * 200, 0.01, 1.0)

        #decide on weather to use total step or just the meta steps for this annealing
        self.eps_u = self.eps_u_decay.value(self.total_steps)
        self.eps_l = self.eps_l_decay.value(self.total_lower_time_step)

        # for storing resutls
        self.results_dict = {
            "train_rewards" : [],
            "train_constraints" : [],
            "eval_rewards" : [],
            "eval_constraints" : [],
        }

        self.cost_indicator = "none"
        if "grid" in self.args.env_name:
            self.cost_indicator = 'pit'
        else:
            raise Exception("not implemented yet")

    def pi_meta(self, state, greedy_eval=False):
        """
        choose goal based on the current policy
        """
        with torch.no_grad():
            # to choose random goal or not
            if (random.random() > self.eps_u_decay.value(self.total_steps)) or greedy_eval:
                q_value = self.dqn_meta(state)
                # chose the max/greedy actions
                goal = q_value.max(1)[1].cpu().numpy()
            else:
                goal = np.random.randint(0, high=self.goal_dim, size = (self.args.num_envs, ))

        return goal

    def pi_lower(self, state, goal, greedy_eval=False):
        """
        take the action based on the current policy
        """
        state_goal = torch.cat((state, goal), dim=1)

        with torch.no_grad():
            # to take random action or not
            if (random.random() > self.eps_l_decay.value(self.total_lower_time_steps)) or greedy_eval:
                q_value = self.dqn_lower(state_goal)
                # chose the max/greedy actions
                action = q_value.max(1)[1].cpu().numpy()
            else:
                action = np.random.randint(0, high=self.action_dim, size = (self.args.num_envs, ))
        return action


    def compute_n_step_returns(self, next_value, rewards, masks):
        """
        n-step SARSA returns
        """
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.args.gamma * R * masks[step]
            returns.insert(0, R)

        return returns

    def run(self):
        """
        Learning happens here
        """
        self.total_steps = 0
        self.total_lower_time_steps = 0
        self.total_meta_time_steps = 0
        self.eval_steps = 0

        # reset state and env
        # reset exploration porcess
        state = self.envs.reset()
        prev_state = state

        ep_reward = 0 #R
        ep_len = 0
        ep_constraint = 0
        start_time = time.time()

        while self.num_episodes < self.args.num_episodes:

            values_u      = []

            states_u      = []
            actions_u     = []
            prev_states_u = []

            rewards     = []
            done_masks  = []
            constraints = []

            for n_u in range(args.traj-len_u):

                state = torch.FloatTensor(state).to(device=self.device)
                goal = self.pi_meta(state=state)

                t_lower = 0

                F = 0


                while t_lower <=  args.max-ep-len_l:
                    instrinc_rewards = []  # for low level n-step
                    values_lower     = []
                    done_masks_lower = []
                    for n_l in range(args.traj-len_u):
                        action = self.pi_lower(state=state, goal=goal)
                        next_state, reward, done, info = self.envs.step(actions=action)
                        instrinc_reward = self.G.intrisic_reward(current_state=next_state,
                                                                 goal_state=goal)




                        q_values = self.dqn_lower(state)
                        Q_value = q_values.gather(1, action)


                        values_lower.append(Q_value)
                        instrinc_rewards.append(torch.FloatTensor(instrinc_reward).unsqueeze(1).to(device=self.device))
                        done_masks_lower.append(torch.FloatTensor(1 - done).unsqueeze(1).to(self.device))


                    t_lower += 1