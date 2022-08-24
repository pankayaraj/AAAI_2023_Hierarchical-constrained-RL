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


from common.past.utils import *

from common.past.multiprocessing_envs import SubprocVecEnv
from torchvision.transforms import ToTensor


from models.past.grid_model import OneHotDQN


from common.past.schedules import LinearSchedule, ExponentialSchedule

class SarsaAgent(object):

    def __init__(self,
                 args,
                 env,
                 writer = None,
                 save_dir=None,
                 exp_no=None
                 ):
        """
        init the agent here
        """

        self.r_path = save_dir + "r" + exp_no
        self.c_path = save_dir + "c" + exp_no

        self.EVAL_REWARDS = []
        self.EVAL_CONSTRAINTS = []

        self.eval_env = copy.deepcopy(env)
        self.args = args

        self.state_dim = env.reset().shape
        self.action_dim = env.action_space.n

        self.device = torch.device("cuda" if (torch.cuda.is_available() and  self.args.gpu) else "cpu")
        print(self.args.gpu)
        # set the same random seed in the main launcher
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        if self.args.gpu:
            torch.cuda.manual_seed(self.args.seed )

        self.writer = writer
        print(self.args.env_name)
        if self.args.env_name == "grid" or self.args.env_name == "grid_key" or self.args.env_name == "four_rooms" or self.args.env_name == "puddle" :
            self.dqn = OneHotDQN(self.state_dim, self.action_dim).to(self.device)
            self.dqn_target = OneHotDQN(self.state_dim, self.action_dim).to(self.device)
        else:
            raise Exception("not implemented yet!")

        # copy parameters
        self.dqn_target.load_state_dict(self.dqn.state_dict())

        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.args.lr)

        # for actors
        def make_env():
            def _thunk():
                env = create_env(args)
                return env

            return _thunk

        #envs = [make_env() for i in range(self.args.num_envs)]
        #self.envs = SubprocVecEnv(envs)
        self.env = env

        # create epsilon and beta schedule
        # NOTE: hardcoded for now
        self.eps_decay = LinearSchedule(50000 * 200, 0.01, 1.0)
        # self.eps_decay = LinearSchedule(self.args.num_episodes * 200, 0.01, 1.0)

        self.total_steps = 0
        self.num_episodes = 0

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
        elif "four_rooms" in self.args.env_name:
            self.cost_indicator = 'pit'
        elif "puddle" in self.args.env_name:
            self.cost_indicator = 'pit'
        else:
            raise Exception("not implemented yet")

        self.eps = self.eps_decay.value(self.total_steps)


    def pi(self, state, greedy_eval=False):
        """
        take the action based on the current policy
        """
        with torch.no_grad():
            # to take random action or not
            if (random.random() > self.eps_decay.value(self.total_steps)) or greedy_eval:
                if state.shape[0] != 1:
                    state = state.unsqueeze(0)

                q_value = self.dqn(state)

                # chose the max/greedy actions

                action = np.array([q_value.max(1)[1].cpu().numpy()])
                action = np.reshape(action,newshape = (self.args.num_envs, ))


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


    def log_episode_stats(self, ep_reward, ep_constraint):
        """
        log the stats for environment performance
        """
        # log episode statistics
        self.results_dict["train_rewards"].append(ep_reward)
        self.results_dict["train_constraints"].append(ep_constraint)
        if self.writer:
            self.writer.add_scalar("Return", ep_reward, self.num_episodes)
            self.writer.add_scalar("Constraint",  ep_constraint, self.num_episodes)


        log(
            'Num Episode {}\t'.format(self.num_episodes) + \
            'E[R]: {:.2f}\t'.format(ep_reward) +\
            'E[C]: {:.2f}\t'.format(ep_constraint) +\
            'avg_train_reward: {:.2f}\t'.format(np.mean(self.results_dict["train_rewards"][-100:])) +\
            'avg_train_constraint: {:.2f}\t'.format(np.mean(self.results_dict["train_constraints"][-100:]))
            )

    def run(self):
        """
        Learning happens here
        """

        self.total_steps = 0
        self.eval_steps = 0

        # reset state and env
        # reset exploration porcess
        #state = self.envs.reset()
        state = self.env.reset()
        prev_state = state

        ep_reward = 0
        ep_len = 0
        ep_constraint = 0
        start_time = time.time()


        while self.num_episodes < self.args.num_episodes:

            values      = []
            c_q_vals    = []
            c_r_vals    = []

            states      = []
            actions     = []
            mus         = []
            prev_states = []

            rewards     = []
            done_masks  = []
            begin_masks = []
            constraints = []


            # n-step sarsa
            for _ in range(self.args.traj_len):

                state = torch.FloatTensor(state).to(self.device)

                # get the action
                action = self.pi(state)
                next_state, reward, done, info = self.env.step(action) #self.envs.step(action)

                # convert it back to tensor
                action = torch.LongTensor(action).unsqueeze(1).to(self.device)


                q_values = self.dqn(state)
                Q_value = q_values.gather(0, action[0])

                # logging mode for only agent 1
                ep_reward += reward

                ep_constraint += info[self.cost_indicator]

                values.append(Q_value)
                rewards.append(reward)
                done_masks.append((1 - done))
                begin_masks.append(info["begin"])
                constraints.append(info[self.cost_indicator])
                prev_states.append(prev_state)
                states.append(state)
                actions.append(action)

                # update states
                prev_state = state
                state = next_state

                self.total_steps += (1 * self.args.num_envs)

                # hack to reuse the same code
                # iteratively add each done episode, so that can eval at regular interval
                if done:
                    if self.num_episodes % self.args.log_every == 0:
                        self.log_episode_stats(ep_reward, ep_constraint)

                    # reset the rewards anyways
                    ep_reward = 0
                    ep_constraint = 0

                    state = self.env.reset()  # reset
                    state = torch.FloatTensor(state).to(device=self.device)

                    self.num_episodes += 1


                    # eval the policy here after eval_every steps
                    if self.num_episodes  % self.args.eval_every == 0:
                        eval_reward, eval_constraint = self.eval()

                        self.EVAL_REWARDS.append(eval_reward)
                        self.EVAL_CONSTRAINTS.append(eval_constraint)

                        torch.save(self.EVAL_REWARDS, self.r_path)
                        torch.save(self.EVAL_CONSTRAINTS, self.c_path)

                        self.results_dict["eval_rewards"].append(eval_reward)
                        self.results_dict["eval_constraints"].append(eval_constraint)

                        log('----------------------------------------')
                        log('Eval[R]: {:.2f}\t'.format(eval_reward) +\
                            'Eval[C]: {}\t'.format(eval_constraint) +\
                            'Episode: {}\t'.format(self.num_episodes) +\
                            'avg_eval_reward: {:.2f}\t'.format(np.mean(self.results_dict["eval_rewards"][-10:])) +\
                            'avg_eval_constraint: {:.2f}\t'.format(np.mean(self.results_dict["eval_constraints"][-10:]))
                            )
                        log('----------------------------------------')

                        if self.writer:
                            self.writer.add_scalar("eval_reward", eval_reward, self.eval_steps)
                            self.writer.add_scalar("eval_constraint", eval_constraint, self.eval_steps)

                        self.eval_steps += 1




            # break here
            if self.num_episodes >= self.args.num_episodes:
                break

            # calculate targets here
            next_state = torch.FloatTensor(next_state).to(self.device)

            next_q_values = self.dqn(next_state)
            next_action = self.pi(next_state)
            next_action = torch.LongTensor(next_action).unsqueeze(1).to(self.device)

            next_q_values = next_q_values.gather(0,next_action[0])


            # calculate targets
            target_Q_vals = self.compute_n_step_returns(next_q_values, rewards, done_masks)
            Q_targets = torch.cat(target_Q_vals).detach()
            Q_values = torch.cat(values)

            # bias corrected loss
            loss  = F.mse_loss(Q_values, Q_targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()



        # done with all the training

        # save the models
        self.save_models()



    def eval(self):
        """
        evaluate the current policy and log it
        """
        avg_reward = []
        avg_constraint = []

        CS = []

        with torch.no_grad():
            for _ in range(self.args.eval_n):

                state = self.eval_env.reset()
                #print(state, self.eval_env.start, self.eval_env.goal)
                done = False
                ep_reward = 0
                ep_constraint = 0
                ep_len = 0
                start_time = time.time()

                while not done:

                    # convert the state to tensor
                    state_tensor =  torch.FloatTensor(state).to(self.device).unsqueeze(0)

                    # get the policy action
                    action = self.pi(state_tensor, greedy_eval=True)
                    next_state, reward, done, info = self.eval_env.step(action)
                    ep_reward += reward
                    ep_len += 1
                    ep_constraint += info[self.cost_indicator]

                    # update the state
                    state = next_state

                    CS.append(next_state)


                avg_reward.append(ep_reward)
                avg_constraint.append(ep_constraint)

        print(CS)

        return np.mean(avg_reward), np.mean(avg_constraint)

    def save_models(self):
        """create results dict and save"""
        models = {
            "dqn" : self.dqn.state_dict(),
            "env" : copy.deepcopy(self.eval_env),
        }
        torch.save(models,os.path.join(self.args.out, 'models.pt'))
        torch.save(self.results_dict, os.path.join(self.args.out, 'results_dict.pt'))


    def load_models(self):
        models = torch.load(os.path.join(self.args.out, 'models.pt'))
        self.dqn.load_state_dict(models["dqn"])
        self.eval_env = models["env"]
