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

from models.ours.grid_model import OneHotDQN

from common.past.schedules import LinearSchedule, ExponentialSchedule

class Dummy(object):

    def __init__(self,
                 args,
                 env,
                 goal_space=None,
                 writer = None,
                 save_dir=None,
                 exp_no=None
                 ):
        """
        init the agent here
        """
        args.num_envs = 1
        if goal_space == None:
            goal_space = args.goal_space
        else:
            raise Exception("Must Specify the goal space as a list")

        self.r_path = save_dir + "r" + exp_no
        self.c_path = save_dir + "c" + exp_no

        self.EVAL_REWARDS = []
        self.EVAL_CONSTRAINTS = []

        self.TRAIN_REWARDS = []
        self.TRAIN_CONSTRAINTS = []



        self.args = args

        #for the time being let's skip the vectorized environment's added complexity in HRL
        self.env = create_env(args)
        self.G = Goal_Space(goal_space=goal_space, grid_size=self.env.size)
        self.grid_size = self.env.size

        c = []
        for i in goal_space:
            c.append(self.G.convert_value_to_coordinates(i))
        print(c)

        self.eval_env = copy.deepcopy(env)

        s = env.reset()
        self.state_dim = s.shape
        self.action_dim = env.action_space.n

        self.goal_dim = self.G.action_shape[0]
        self.goal_state_dim = np.concatenate((s,s)).shape

        # these are the cost conditioned value functions that will be used for reward Q vlaue functions
        self.cost_goal_state_dim = (self.goal_state_dim[0] + 1,)
        self.cost_state_dim = (self.state_dim[0] + 1,)

        self.device = torch.device("cuda" if (torch.cuda.is_available() and  self.args.gpu) else "cpu")

        # set the same random seed in the main launcher
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        if self.args.gpu:
            torch.cuda.manual_seed(self.args.seed )

        self.writer = writer

        if self.args.env_name == "grid" or self.args.env_name == "grid_key":
            self.dqn_meta = OneHotDQN(self.cost_state_dim, self.goal_dim).to(self.device)
            self.dqn_meta_target = OneHotDQN(self.cost_state_dim, self.goal_dim).to(self.device)

            self.dqn_lower = OneHotDQN(self.cost_goal_state_dim, self.action_dim).to(self.device)
            self.dqn_lower_target = OneHotDQN(self.cost_goal_state_dim, self.action_dim).to(self.device)
        else:
            raise Exception("not implemented yet!")

        # copy parameters
        self.dqn_meta_target.load_state_dict(self.dqn_meta.state_dict())
        self.dqn_lower_target.load_state_dict(self.dqn_lower.state_dict())

        self.optimizer_meta = torch.optim.Adam(self.dqn_meta.parameters(), lr=self.args.lr)
        self.optimizer_lower = torch.optim.Adam(self.dqn_lower.parameters(), lr=self.args.lr)

        """
        def make_env():
            def _thunk():
                env = create_env(args)
                return env

            return _thunk

        envs = [make_env() for i in range(self.args.num_envs)]
        self.envs = SubprocVecEnv(envs)
        """



        self.total_steps = 0
        self.total_lower_time_steps = 0
        self.total_meta_time_steps = 0
        self.num_episodes = 0

        #different epsilon for different levels
        self.eps_u_decay = LinearSchedule(1000000 * 200, 0.01, 1.0)
        self.eps_l_decay = LinearSchedule(50000 * 200, 0.01, 1.0)

        #decide on weather to use total step or just the meta steps for this annealing
        self.eps_u = self.eps_u_decay.value(self.total_steps)
        self.eps_l = self.eps_l_decay.value(self.total_lower_time_steps)

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

    def pi_meta(self, state, cost, greedy_eval=False):
        """
        choose goal based on the current policy
        """
        state_cost = torch.cat((state, cost))
        with torch.no_grad():

            self.eps_u = self.eps_u_decay.value(self.total_steps)
            # to choose random goal or not
            if (random.random() > self.eps_u) or greedy_eval:
                q_value = self.dqn_meta(state_cost)

                # chose the max/greedy actions
                goal = np.array([q_value.max(0)[1].cpu().numpy()])
            else:
                goal = np.random.randint(0, high=self.goal_dim, size = (self.args.num_envs, ))

        return goal

    def pi_lower(self, state, goal, cost, greedy_eval=False):
        """
        take the action based on the current policy
        """
        state_goal_cost = torch.cat((torch.cat((state, goal)), cost))

        self.eps_l = self.eps_l_decay.value(self.total_lower_time_steps)
        with torch.no_grad():
            # to take random action or not
            if (random.random() > self.eps_l) or greedy_eval:
                q_value = self.dqn_lower(state_goal_cost)
                # chose the max/greedy actions
                action = np.array([q_value.max(0)[1].cpu().numpy()])

                #print(action)
                #print("action_greedy")
            else:
                action = np.random.randint(0, high=self.action_dim, size = (self.args.num_envs, ))
                #print(action)
                #print("action_random")
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
        self.TRAIN_REWARDS.append(ep_reward)
        self.TRAIN_CONSTRAINTS.append(ep_constraint)



        log(
            'Num Episode {}\t'.format(self.num_episodes) + \
            'E[R]: {:.2f}\t'.format(ep_reward) +\
            'E[C]: {:.2f}\t'.format(ep_constraint) +\
            'avg_train_reward: {:.2f}\t'.format(np.mean(self.TRAIN_REWARDS[-100:])) +\
            'avg_train_constraint: {:.2f}\t'.format(np.mean(self.TRAIN_CONSTRAINTS[-100:]))
            )

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
        state = self.env.reset()
        state = torch.FloatTensor(state).to(device=self.device)

        previous_state = state
        current_cost = torch.zeros(self.args.num_envs, 1).float().to(self.device)
        cost = torch.zeros(self.args.num_envs).to(self.device)
        cost[0] = self.args.d0

        #total episode reward, length for logging purposes
        self.ep_reward = 0
        self.ep_len = 0
        self.ep_constraint = 0
        start_time = time.time()


        while self.num_episodes < self.args.num_episodes:


            next_state = None
            done = None

            states_u      = []
            actions_u     = []
            prev_states_u = []

            rewards     = []
            done_masks  = []
            constraints = []

            IR_t = []
            Goals_t = []
            CS_t = []
            T_t = []

            t_upper = 0
            while not done:
                values_upper = []
                rewards_upper = []
                done_masks = []



                t_upper += 1

                previous_state = state

                state_cost = torch.cat((state, cost))

                goal = self.pi_meta(state=state, cost=cost)

                x_g, y_g = self.G.convert_value_to_coordinates(self.G.goal_space[goal.item()])
                Goals_t.append((x_g, y_g, self.G.goal_space[goal.item()]))

                goal = torch.LongTensor(goal).unsqueeze(1).to(self.device)

                q_values_upper = self.dqn_meta(state_cost)
                Q_value_upper = q_values_upper.gather(0, goal[0])

                #an indicator that is used to terminate the lower level episode
                t_lower = 0

                eps_reward_lower = 0
                R = 0

                goal_hot_vec = self.G.covert_value_to_hot_vec(goal)
                goal_hot_vec = torch.FloatTensor(goal_hot_vec).to(self.device)

                L = None
                #this will terminate of the current lower level episoded went beyond limit

                while t_lower <= self.args.max_ep_len_l-1:

                    instrinc_rewards = []  # for low level n-step
                    values_lower     = []
                    done_masks_lower = []
                    for n_l in range(self.args.traj_len_l):
                        action = self.pi_lower(state=state, goal=goal_hot_vec, cost=cost)

                        state_goal_cost = torch.cat((torch.cat((state, goal_hot_vec)), cost))

                        next_state, reward, done, info = self.env.step(action=action.item())

                        instrinc_reward = self.G.intrisic_reward(current_state=next_state,
                                                                 goal_state=goal_hot_vec)

                        next_state = torch.FloatTensor(next_state).to(self.device)
                        done_l = self.G.validate(current_state=next_state, goal_state=goal_hot_vec)  #this is to validate the end of the lower level episode


                        action = torch.LongTensor(action).unsqueeze(1).to(self.device)


                        R += reward

                        #for training logging purposes
                        self.ep_len += 1
                        self.ep_constraint += info[self.cost_indicator]
                        self.ep_reward += reward



                        state_goal = torch.cat((state, goal_hot_vec))

                        q_values_lower = self.dqn_lower(state=state_goal_cost)
                        Q_value_lower = q_values_lower.gather(0, action[0])


                        values_lower.append(Q_value_lower)
                        instrinc_rewards.append(instrinc_reward)
                        done_masks_lower.append((1 -done_l))

                        t_lower += 1
                        self.total_steps += 1
                        self.total_lower_time_steps += 1

                        state = next_state

                        #break if goal is current_state or the if the main episode terminated

                        if done or done_l:
                            break

                        if t_lower > self.args.max_ep_len_l-1:

                            break



                    x_c, y_c = self.G.convert_value_to_coordinates(self.G.convert_hot_vec_to_value(next_state).item())


                    next_state_goal = torch.cat((next_state, goal_hot_vec))
                    next_state_goal_cost = torch.cat((torch.cat((next_state, goal_hot_vec)), cost))

                    next_action = self.pi_lower(next_state, goal_hot_vec, cost)
                    next_action = torch.LongTensor(next_action).unsqueeze(1).to(self.device)
                    next_values = self.dqn_lower(next_state_goal_cost)
                    Next_Value = next_values.gather(0, next_action[0])


                    target_Q_values_lower = self.compute_n_step_returns(Next_Value, instrinc_rewards, done_masks_lower)
                    Q_targets_lower = torch.cat(target_Q_values_lower).detach()
                    Q_values_lower = torch.cat(values_lower)

                    loss_lower = F.mse_loss(Q_values_lower, Q_targets_lower)

                    self.optimizer_lower.zero_grad()
                    loss_lower.backward()
                    self.optimizer_lower.step()

                    if done:
                        break

                values_upper.append(Q_value_upper)
                rewards_upper.append(R)
                done_masks.append((1 - done))

                CS_t.append((x_c, y_c))
                T_t.append(t_lower)

                next_state_cost = torch.cat((next_state, cost))

                next_goal = self.pi_meta(next_state, cost)
                next_goal = torch.LongTensor(next_goal).unsqueeze(1).to(self.device)
                next_values = self.dqn_meta(next_state_cost)
                Next_Value = next_values.gather(0, next_goal[0])

                target_Q_values_upper = self.compute_n_step_returns(Next_Value, rewards_upper, done_masks)
                Q_targets_upper = torch.cat(target_Q_values_upper).detach()
                Q_values_upper = torch.cat(values_upper)

                loss_upper = F.mse_loss(Q_values_upper, Q_targets_upper)
                self.optimizer_meta.zero_grad()
                loss_upper.backward()
                self.optimizer_meta.step()

                if done:

                    self.num_episodes += 1

                    #training logging
                    if self.num_episodes % 100 == 0:
                        self.log_episode_stats(ep_reward=self.ep_reward, ep_constraint=self.ep_constraint)

                    if self.num_episodes % 500 == 0:
                        log("Goal State: "  + " " + str(Goals_t) + " Current State: " + str(CS_t))
                        log("No of Higher Eps: " +  str(len(Goals_t)) + " No of lower eps: " + str(T_t))

                    #evaluation logging
                    if self.num_episodes % self.args.eval_every == 0:
                        eval_reward, eval_constraint, IR, Goals, CS = self.eval()

                        print("Epsilon Upper and Lower:" + str(self.eps_u) +", " + str(self.eps_l))

                        self.EVAL_REWARDS.append(eval_reward)
                        self.EVAL_CONSTRAINTS.append(eval_constraint)

                        torch.save(self.EVAL_REWARDS, self.r_path)
                        torch.save(self.EVAL_CONSTRAINTS, self.c_path)

                        log('----------------------------------------')
                        log("Intrisic Reward: " + str(IR) + " Goal: " + str(Goals) + " Current State: " + str(CS))
                        log('Eval[R]: {:.2f}\t'.format(eval_reward) + \
                            'Eval[C]: {}\t'.format(eval_constraint) + \
                            'Episode: {}\t'.format(self.num_episodes) + \
                            'avg_eval_reward: {:.2f}\t'.format(np.mean(self.EVAL_REWARDS[-10:])) + \
                            'avg_eval_constraint: {:.2f}\t'.format(np.mean(self.EVAL_CONSTRAINTS[-10:]))
                            )
                        log('----------------------------------------')

                    # resting episode rewards
                    self.ep_reward = 0
                    self.ep_len = 0
                    self.ep_constraint = 0

                    state = self.env.reset() #reset
                    state = torch.FloatTensor(state).to(device=self.device)
                    break #this break is to terminate the higher tier episode as the episode is now over




    def eval(self):
        """
                    evaluate the current policy and log it
                    """
        avg_reward = []
        avg_constraint = []

        cost = torch.zeros(self.args.num_envs).to(self.device)
        cost[0] = self.args.d0

        with torch.no_grad():
            for _ in range(self.args.eval_n):

                state = self.eval_env.reset()
                previous_state = torch.FloatTensor(state)
                done = False
                ep_reward = 0
                ep_constraint = 0
                ep_len = 0
                start_time = time.time()

                IR = []
                Goals = []
                CS = []
                while not done:

                    # convert the state to tensor
                    state = torch.FloatTensor(state).to(self.device)

                    # get the goal
                    goal = self.pi_meta(state, cost, greedy_eval=True)

                    x_g, y_g = self.G.convert_value_to_coordinates(self.G.goal_space[goal.item()])
                    Goals.append((x_g, y_g))

                    goal = torch.LongTensor(goal).unsqueeze(1).to(self.device)


                    goal_hot_vec = self.G.covert_value_to_hot_vec(goal)
                    goal_hot_vec = torch.FloatTensor(goal_hot_vec)

                    t_lower = 0
                    ir = 0
                    while t_lower <= self.args.max_ep_len_l:

                        action = self.pi_lower(state, goal_hot_vec, cost, greedy_eval=True)
                        #print(torch.equal(state, previous_state), self.G.convert_hot_vec_to_value(state), self.G.convert_hot_vec_to_value(goal_hot_vec))
                        #print(self.dqn_lower(torch.cat((state, goal_hot_vec))), t_lower)

                        next_state, reward, done, info = self.eval_env.step(action.item())
                        ep_reward += reward
                        ep_len += 1
                        ep_constraint += info[self.cost_indicator]

                        """
                        NS = []
                        for i in range(4):
                            NS.append(self.eval_env.step(i)[0])

                        T = []
                        for ts in NS:
                            T.append(torch.equal(state, torch.FloatTensor(ts)))
                        print(T)
                        """
                        next_state = torch.FloatTensor(next_state).to(self.device)

                        # update the state
                        previous_state = state
                        state = next_state

                        instrinc_reward = self.G.intrisic_reward(current_state=next_state,
                                                                 goal_state=goal_hot_vec)
                        ir += instrinc_reward
                        t_lower += 1

                        done_l = self.G.validate(current_state=next_state, goal_state=goal_hot_vec)
                        if done_l or done:
                            break

                    IR.append(ir)

                    x_c, y_c = self.G.convert_value_to_coordinates(self.G.convert_hot_vec_to_value(next_state).item())

                    CS.append((x_c, y_c))

                avg_reward.append(ep_reward)
                avg_constraint.append(ep_constraint)



        return np.mean(avg_reward), np.mean(avg_constraint), IR, Goals, CS



