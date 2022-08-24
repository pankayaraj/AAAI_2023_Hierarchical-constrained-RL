import gym
from gym import core, spaces
from gym.envs.registration import register
import numpy as np
from gym.utils import seeding
import copy

#adapted from the environemnt at https://github.com/arushijain94/SafeOptionCritic

class PuddleSimpleEnv(gym.Env):

    def __init__(self, max_steps=100, noise=0.025, thrust=0.05, puddle_center=[[.5, .5]],
            puddle_width=[[.3, .3]]):


        self.size = 1.0

        goal = [0.1, 1.0]
        goal_threshold = 0.1
        start = [0.0, 0.0]
        self.max_steps = max_steps

        #main cost
        cost1 = [[0.0, 0.3], [0.0, 0.6], [0.7, 0.3], [0.7, 0.6]]
        cost2 = [[0.9, 0.0], [0.9, 0.2], [1.0, 0.0], [1.0, 0.2]]
        cost3 = [[0.5, 0.9],  [0.5, 1.0], [0.6, 0.9], [0.6, 1.0]]

        #cost indicated by bottom left and top right point
        self.cost1 = [[0.7, 0.3], [0.0, 0.6]]
        self.cost2 = [[1.0, 0.0], [0.85, 0.5]]
        self.cost3 = [[0.6, 0.9], [0.5, 1.0]]

        self.start = np.array(start)
        self.goal = np.array(goal)
        self.goal_threshold = goal_threshold
        self.thrust = thrust

        self.noise = noise
        self.puddle_center = [np.array(center) for center in puddle_center]
        self.puddle_width = [np.array(width) for width in puddle_width]


        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2,))

        self.actions = [np.zeros(2) for i in range(4)]
        for i in range(4):
            self.actions[i][i//2] = thrust * (i%2 * 2 - 1)

        self._seed()
        self.viewer = None
        self.reset()


    def isinthebox(self, cost, point):
        bl = cost[0]
        tr = cost[1]
        p  = point

        if (p[0] < bl[0] and p[0] > tr[0] and p[1] > bl[1] and p[1] < tr[1]):
            return True
        else:
            return False

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        self.t += 1
        action = action
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))


        self.pos += self.actions[action] + self.np_random.uniform(low=-self.noise, high=self.noise, size=(2,))
        self.pos = np.clip(self.pos, 0.0, 1.0)




        reward = 0.
        cost   = 0.

        done = np.linalg.norm((self.pos - self.goal), ord=1) < self.goal_threshold


        if np.linalg.norm((self.pos - self.goal), ord=1) < self.goal_threshold:
            reward = 1000.0

        if self.t >= self.max_steps:
            done = True

        if self.isinthebox(self.cost1, self.pos) or self.isinthebox(self.cost1, self.pos) or self.isinthebox(self.cost1, self.pos):
            cost  = 10.0

        info = {}
        info['pit'] = cost
        if self.t == 1:
            info['begin'] = True
        else:
            info['begin'] = False

        return np.clip(self.pos, 0.0, 1.0), reward, done, info

    def reset(self):

        self.t = 0
        self.pos = np.array([0.0, 0.0])
        return self.pos


P = PuddleSimpleEnv()

print(P.isinthebox(P.cost1, np.array([0.2, 0.2])))
