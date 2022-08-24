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

from torch.utils.tensorboard import SummaryWriter

from common.past.utils import *

from common.past.multiprocessing_envs import SubprocVecEnv
from torchvision.transforms import ToTensor


from models.past.grid_model import DQN, OneHotDQN, miniDQN, OneHotValueNetwork


from common.past.schedules import LinearSchedule, ExponentialSchedule

class SafeSarsaAgent(object):

    def __init__(self,
                 args,
                 env,
                 writer = None,
                 save_dir = None,
                 exp_no = None):
        """
        init agent
        """
        self.args = args
        self.args.cost_reverse_lr = 5e-5
        self.args.cost_q_lr = 5e-5

        self.cost_estimate_L = []
        self.action_mask_L = []
        self.cost_L = []

        self.f_q_value = []
        self.r_value = []


        self.save_dir = save_dir
        self.exp_no = exp_no

        self.r_path = save_dir + "r" +  exp_no
        self.c_path = save_dir + "c" + exp_no

        self.EVAL_REWARDS = []
        self.EVAL_CONSTRAINTS = []


        self.eval_env = copy.deepcopy(env)


        self.state_dim = env.reset().shape

        self.action_dim = env.action_space.n

        self.device = torch.device("cuda" if (torch.cuda.is_available() and  self.args.gpu) else "cpu")

        # set the random seed in the main launcher
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        if self.args.gpu:
            torch.cuda.manual_seed(self.args.seed )

        self.writer = writer

        if self.args.env_name == "grid" or self.args.env_name == "grid_key" or self.args.env_name == "four_rooms" or self.args.env_name == "puddle" :
            self.dqn = OneHotDQN(self.state_dim, self.action_dim).to(self.device)
            self.dqn_target = OneHotDQN(self.state_dim, self.action_dim).to(self.device)

            # create more networks here
            self.cost_model = OneHotDQN(self.state_dim, self.action_dim).to(self.device)
            self.review_model = OneHotValueNetwork(self.state_dim).to(self.device)

            self.target_cost_model = OneHotDQN(self.state_dim, self.action_dim).to(self.device)
            self.target_review_model = OneHotValueNetwork(self.state_dim).to(self.device)

            self.target_cost_model.load_state_dict(self.cost_model.state_dict())
            self.target_review_model.load_state_dict(self.review_model.state_dict())
        else:
            raise Exception("Not implemented yet!")

        # copy parameters
        self.dqn_target.load_state_dict(self.dqn.state_dict())

        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.args.lr)
        self.review_optimizer = optim.Adam(self.review_model.parameters(), lr=self.args.cost_reverse_lr)
        self.critic_optimizer = optim.Adam(self.cost_model.parameters(), lr=self.args.cost_q_lr)

        # parallel agents
        def make_env():
            def _thunk():
                env = create_env(args)
                return env

            return _thunk

        #envs = [make_env() for i in range(self.args.num_envs)]
        #self.envs = SubprocVecEnv(envs)
        self.env = env


        # create epsilon  and beta schedule
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



    def pi(self, state, current_cost=0.0, greedy_eval=False):
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
                action = np.reshape(action, newshape=(self.args.num_envs,))

            else:
                action = np.random.randint(0, high=self.action_dim, size = (self.args.num_envs, ))

        return action

    def safe_deterministic_pi(self, state,  current_cost=0.0, greedy_eval=False):
        """
        take the action based on the current policy
        """

        with torch.no_grad():
            # to take random action or not
            if (random.random() > self.eps_decay.value(self.total_steps)) or greedy_eval:
                # No random action
                if state.shape[0] != 1:
                    state = state.unsqueeze(0)

                q_value = self.dqn(state)

                # Q_D(s,a)
                cost_q_val = self.cost_model(state)
                cost_r_val = self.review_model(state)


                # find the action set
                # create the filtered mask here
                constraint_mask = torch.le(cost_q_val + cost_r_val, self.args.d0 + current_cost).float()


                filtered_Q = (q_value + 1000.0) * (constraint_mask)

                filtered_action = np.array([filtered_Q.max(1)[1].cpu().numpy()])
                filtered_action = np.reshape(filtered_action, newshape=(self.args.num_envs,))
                # alt action to take if infeasible solution
                # minimize the cost


                alt_action = np.array([(-1. * cost_q_val).max(1)[1].cpu().numpy()])
                alt_action = np.reshape(alt_action, newshape=(self.args.num_envs,))

                c_sum = constraint_mask.sum(1)
                action_mask = ( c_sum == torch.zeros_like(c_sum)).cpu().numpy()

                action = (1 - action_mask) * filtered_action + action_mask * alt_action

                cost_estimate = cost_q_val + cost_r_val - current_cost

                return action, cost_estimate, action_mask
            else:
                # create an array of random indices, for all the environments
                action = np.random.randint(0, high=self.action_dim, size = (self.args.num_envs, ))

                return action, None, None


    def set_policy(self):
        self.policy = self.safe_deterministic_pi

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

    def compute_reverse_n_step_returns(self, prev_value, costs, begin_masks):
        """
        n-step SARSA returns (backward in time)
        """
        R = prev_value
        returns = []
        for step in range(len(costs)):
            R = costs[step] + self.args.gamma * R * begin_masks[step]
            returns.append(R)

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
        learning happens here
        """
        self.set_policy()

        self.total_steps = 0
        self.eval_steps = 0

        # reset state and env
        state = self.env.reset()
        prev_state = torch.FloatTensor(state).to(self.device)
        current_cost = torch.zeros(self.args.num_envs,1).float().to(self.device)

        ep_reward = 0
        ep_len = 0
        ep_constraint = 0
        start_time = time.time()


        psg = state
        done = False
        while self.num_episodes < self.args.num_episodes:

            C = 0

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

            eps_cost_estimate_l = []
            eps_action_mask_l = []
            eps_cost_l = []

            eps_f_q_value = []
            eps_r_value = []

            # n-step sarsa
            for _ in range(self.args.traj_len):

                state = torch.FloatTensor(state).to(self.device)

                # get the action
                action, cost_estimate, action_mask = self.policy(state, current_cost= current_cost)
                next_state, reward, done, info = self.env.step(action)




                # convert it back to tensor
                action = torch.LongTensor(action).unsqueeze(1).to(self.device)

                q_values = self.dqn(state)
                Q_value = q_values.gather(0, action[0])

                c_q_values = self.cost_model(state)
                cost_q_val = c_q_values.gather(0, action[0])

                cost_r_val = self.review_model(state)

                # logging mode for only agent 1
                ep_reward += reward
                ep_constraint += info[self.cost_indicator]

                C += info[self.cost_indicator]

                eps_cost_estimate_l.append(cost_estimate)
                eps_action_mask_l.append(action_mask)

                eps_cost_l.append(C)
                eps_f_q_value.append(cost_q_val.item())
                eps_r_value.append(cost_r_val.item())

                values.append(Q_value)
                c_r_vals.append(cost_r_val)
                c_q_vals.append(cost_q_val)
                rewards.append(reward)
                done_masks.append((1.0 - done))
                begin_masks.append((1.0 - info['begin']))
                constraints.append(info[self.cost_indicator])

                prev_states.append(prev_state)
                states.append(state)
                actions.append(action)

                # update the state
                prev_state = state
                state = next_state

                # update the current cost
                # if done flag is true for the current env, this implies that the next_state cost = 0.0
                # because the agent starts with 0.0 cost (or doesn't have access to it anyways)
                current_cost = torch.FloatTensor([info[self.cost_indicator] * (1.0 - done)]).unsqueeze(1).to(self.device)

                self.total_steps += (1 * self.args.num_envs)

                # hack to reuse the same code
                # iteratively add each done episode, so that can eval at regular interval
                if done:




                    if self.num_episodes % 100 == 0:
                        self.log_episode_stats(ep_reward, ep_constraint)



                        # reset the rewards anyways
                    ep_reward = 0
                    ep_constraint = 0
                    state = self.env.reset()
                    state = torch.FloatTensor(state).to(device=self.device)

                    self.num_episodes += 1

                    #self.action_mask_L.append(eps_action_mask_l)
                    #self.cost_estimate_L.append(eps_cost_estimate_l)
                    self.cost_L.append(eps_cost_l)
                    self.f_q_value.append(eps_f_q_value)
                    self.r_value.append(eps_r_value)



                    # eval the policy here after eval_every steps
                    if self.num_episodes  % 1000 == 0:

                        p1 = self.save_dir + "cost_estimate_" + self.exp_no
                        p2 = self.save_dir + "cost_" + self.exp_no
                        p3 = self.save_dir + "action_mask" + self.exp_no
                        p4 = self.save_dir + "f_q_value" + self.exp_no
                        p5 = self.save_dir + "r_value" + self.exp_no

                        #torch.save(self.cost_estimate_L, p1)
                        torch.save(self.cost_L, p2)
                        #torch.save(self.action_mask_L, p3)
                        torch.save(self.f_q_value, p4)
                        torch.save(self.r_value, p5)

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

            next_action, cost_estimate, action_mask = self.policy(next_state, current_cost)
            next_action = torch.LongTensor(next_action).unsqueeze(1).to(self.device)
            next_q_values = next_q_values.gather(0, next_action[0])

            # calculate targets
            target_Q_vals = self.compute_n_step_returns(next_q_values, rewards, done_masks)
            Q_targets = torch.cat(target_Q_vals).detach()


            Q_values = torch.cat(values)

            # loss
            loss  = F.mse_loss(Q_values, Q_targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # calculate the cost-targets
            next_c_value = self.cost_model(next_state)
            next_c_value = next_c_value.gather(0, next_action[0])

            cq_targets = self.compute_n_step_returns(next_c_value, constraints, done_masks)
            C_q_targets = torch.cat(cq_targets).detach()
            C_q_vals = torch.cat(c_q_vals)

            cost_critic_loss = F.mse_loss(C_q_vals, C_q_targets)
            self.critic_optimizer.zero_grad()
            cost_critic_loss.backward()
            self.critic_optimizer.step()

            # For the constraints (reverse)
            prev_value = self.review_model(prev_states[0])

            c_r_targets = self.compute_reverse_n_step_returns(prev_value, constraints, begin_masks)
            C_r_targets = torch.cat(c_r_targets).detach()
            C_r_vals = torch.cat(c_r_vals)

            cost_review_loss = F.mse_loss(C_r_vals, C_r_targets)
            self.review_optimizer.zero_grad()
            cost_review_loss.backward()
            self.review_optimizer.step()



        # done with all the training

        # save the models
        self.save_models()



    def eval(self):
        """
        evaluate the current policy and log it
        """
        self.set_policy()

        avg_reward = []
        avg_constraint = []

        with torch.no_grad():
            for _ in range(self.args.eval_n):

                state = self.eval_env.reset()
                done = False
                ep_reward = 0
                ep_constraint = 0
                ep_len = 0
                start_time = time.time()
                current_cost = torch.FloatTensor([0.0]).to(self.device).unsqueeze(0)

                while not done:

                    # convert the state to tensor
                    state =  torch.FloatTensor(state).to(self.device).unsqueeze(0)


                    # get the policy action
                    action, cost_estimate, action_mask = self.policy(state, current_cost=current_cost, greedy_eval=True)
                    #action = action[0]

                    next_state, reward, done, info = self.eval_env.step(action)
                    ep_reward += reward
                    ep_len += 1
                    ep_constraint += info[self.cost_indicator]



                    # update the state
                    state = next_state
                    current_cost = torch.FloatTensor([info[self.cost_indicator] * (1.0 - done)]).to(self.device).unsqueeze(0)



                avg_reward.append(ep_reward)
                avg_constraint.append(ep_constraint)

        return np.mean(avg_reward), np.mean(avg_constraint)

    def save_models(self):
        """create results dict and save"""
        models = {
            "dqn" : self.dqn.state_dict(),
            "cost_model" : self.cost_model.state_dict(),
            "review_model" : self.review_model.state_dict(),
            "env" : copy.deepcopy(self.eval_env),
        }
        torch.save(models,os.path.join(self.args.out, 'models.pt'))

        # save the results
        torch.save(self.results_dict, os.path.join(self.args.out, 'results_dict.pt'))


    def load_models(self):
        models = torch.load(os.path.join(self.args.out, 'models.pt'))
        self.dqn.load_state_dict(models["dqn"])
        self.eval_env = models["env"]
