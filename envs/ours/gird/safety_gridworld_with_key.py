"""

OBTAINED FORM THE WORK: https://github.com/hercky/cmdps_via_bvf

Creates the map as in Safe Lyp RL paper

Main code from here: https://github.com/junhyukoh/value-prediction-network/blob/master/maze.py
And visualization inspired from: https://github.com/WojciechMormul/rl-grid-world
"""


from PIL import Image
import numpy as np
import gym
from gym import spaces
import copy

import torch


# constants
BLOCK = 0
AGENT = 1
GOAL = 2
PIT = 3
KEY = 4

# movemnent, can only move in 4 directions
DX = [0, 1, 0, -1]
DY = [-1, 0, 1, 0]


# for generation purposes
COLOR = [
        [44, 42, 60], # block - black
        [91, 255, 123], # agent - green
        [52, 152, 219], # goal - blue
        [255, 0, 0], # pit - red
        [245, 239, 66], #key - yellow
        ]


def generate_maze(size=27, obstacle_density=0.3, gauss_placement=False, rand_goal=True):
    """
    returns the 4 rooms maze, start and goal
    """
    mx = size-2; my = size-2 # width and height of the maze
    maze = np.zeros((my, mx))

    #NOTE: padding here
    dx = DX
    dy = DY


    # define the start and the goal
    # start in (0, alpha)
    start_y, start_x =  my-1, mx//2-1
    key_y, key_x = my//2-1, mx-1

    # goal position   (24,24)
    if rand_goal:
        goal_y, goal_x = np.random.randint(0,my), 0
    else:
        goal_y, goal_x = my//2, 0


    # create the actual maze here
    # maze_tensor = np.zeros((size, size, len(COLOR)))
    maze_tensor = np.zeros((len(COLOR), size, size))

    # fill everything with blocks
    # maze_tensor[:,:,BLOCK] = 1.0
    maze_tensor[BLOCK,:,:] = 1.0

    # fit the generated maze
    # maze_tensor[1:-1, 1:-1, BLOCK] = maze
    maze_tensor[BLOCK, 1:-1, 1:-1] = maze

    # put the agent
    # maze_tensor[start_y+1][start_x+1][AGENT] = 1.0
    maze_tensor[AGENT][start_y+1][start_x+1]= 1.0

    # put the goal
    # maze_tensor[goal_y+1][goal_x+1][GOAL] = 1.0
    maze_tensor[GOAL][goal_y+1][goal_x+1] = 1.0
    maze_tensor[KEY][key_y + 1][key_x + 1] = 1.0

    # put the pits

    # create the the pits here
    for i in range(0, mx):
        for j in range(0, my):
            # pass if start or goal state
            if (i==start_x and j==start_y) or (i==goal_x and j==goal_y) or (i == key_x and j == key_y): #last condition in this is new as of now u didn't run anything on it so check I guess
            #if (i==start_x and j==start_y) or (i==goal_x and j==goal_y):
                pass

            # with prob p place the pit
            if np.random.rand() < obstacle_density:
                # maze_tensor[j+1][i+1][PIT] = 1.0
                maze_tensor[PIT][j+1][i+1] = 1.0

    #print("Goal: " + str((goal_x+1, goal_y+1)))
    #print("Key: " + str((key_x+1, key_y+1)))
    return maze_tensor, [start_y+1, start_x+1], [goal_y+1, goal_x+1]




class PitWorld_Key(gym.Env):
    """
    the env from safe lyp RL
    """
    def __init__(self,
                 size = 27,
                 max_step = 200,
                 per_step_penalty = -1.0,
                 #per_step_penalty=0.0,
                 goal_reward = 1000.0,
                 obstace_density = 0.3,
                 constraint_cost = 1.0,
                 random_action_prob = 0.005,
                 rand_goal = True,
                 one_hot_features=False):
        """
        create maze here
        """

        self.size = size
        self.dy = DY
        self.dx = DX
        self.random_action_prob = random_action_prob
        self.per_step_penalty = per_step_penalty
        self.goal_reward = goal_reward
        self.obstace_density = obstace_density
        self.max_step = max_step
        self.constraint_cost = constraint_cost
        self.one_hot = one_hot_features
        self.rand_goal = rand_goal

        self.key_picked = False

        # 4 possible actions
        self.action_space = spaces.Discrete(4)

        # create the maze
        self.init_maze, self.start_pos, self.goal_pos = generate_maze(size=self.size,
                                                                      obstacle_density=self.obstace_density,
                                                                      rand_goal = self.rand_goal)

        self.goal_y   = self.goal_pos[0]
        self.goal_x   = self.goal_pos[1]
        self.start_y  = self.start_pos[0]
        self.start_x  = self.start_pos[1]

        # observation space
        # TODO: 4d tensor or 3d image

        if self.one_hot is False:
            self.observation_space = spaces.Box(low=0, high=1,
                                            shape=self.init_maze.shape)
        else:
            self.observation_space = spaces.Box(low=0, high=1,
                                            shape=self.init_maze[AGENT].shape)


        self.reset()

    def reset(self):
        """
        """
        self.key_picked = False

        self.maze = copy.deepcopy(self.init_maze)
        self.agent_pos = copy.deepcopy(self.start_pos)

        self.t = 0
        self.episode_reward = 0
        self.done = False

        return self.observation()


    def observation(self):
        obs = np.array(self.maze, copy=True)

        if self.one_hot is False:
            # returns in the (channel, height, width) format
            obs = np.reshape(obs, (-1, self.size, self.size))
        else:
            obs = obs[AGENT].flatten()


        return obs


    def visualize(self, img_size=320):
        """
        create an image
        """
        img_maze = np.array(self.maze, copy=True).reshape(self.size, self.size, -1)
        #         currently for maze[y][x][color]
        my = self.maze.shape[1]
        mx = self.maze.shape[2]
        colors = np.array(COLOR, np.uint8)


        print(self.maze.shape)
        vis_maze = np.zeros(shape=( mx, my, 3 ))


        for i in range(len(COLOR)):
            for j in range(mx):
                for k in range(my):
                    if self.maze[i][j][k] == 1:

                        vis_maze[j][k] = np.array(COLOR[i])
                    else:
                        vis_maze[j][k] = np.array([255, 255, 255])

        #vis_maze = np.matmul(self.maze, colors[:num_channel])
        #vis_maze = vis_maze.astype(np.uint8)
        #for i in range(vis_maze.shape[0]):
        #    for j in range(vis_maze.shape[1]):
        #        if self.maze[i][j].sum() == 0.0:
        #            vis_maze[i][j][:] = int(255)

        image = Image.fromarray(vis_maze)
        return image.resize((int(float(img_size) * mx / my), img_size), Image.NEAREST)



    def to_string(self, goal_space=None):

        #here goal space is given as a list of integer values as done in the arguments. This it is converted into coordinates as done int \ours\grid\utils.py
        SUB_G = np.zeros((self.size, self.size))
        if goal_space != None:
            g_s_x = []
            g_s_y = []

            for i in goal_space:
                g_s_x.append((i) % self.size)
                g_s_y.append((i) //self.size)

            for j in range(len(goal_space)):
                SUB_G[g_s_y[j]][g_s_x[j]] = 1

        my = self.maze.shape[1]
        mx = self.maze.shape[2]

        maze_grid = [[0 for _ in range(mx)] for __ in range(my)]

        str = ''
        for y in range(my):
            for x in range(mx):
                if self.maze[BLOCK][y][x] == 1:
                    maze_grid[y][x] = 0
                    str += '  #'
                elif self.maze[AGENT][y][x] == 1:
                    str += '  A'
                    maze_grid[y][x] = 1
                elif self.maze[GOAL][y][x] == 1:
                    str += '  G'
                    maze_grid[y][x] = 2
                elif self.maze[PIT][y][x] == 1:
                    str += '  x'
                    maze_grid[y][x] = 3
                elif self.maze[KEY][y][x] == 1:
                    str += '  K'
                    maze_grid[y][x] = 4
                elif goal_space != None:
                    if SUB_G[y][x] == 1:
                        str += '  S'
                        maze_grid[y][x] = 5
                    else:
                        str += '   '
                        maze_grid[y][x] = 6
                else:
                    str += '   '
                    maze_grid[y][x] = 6
            str += '\n'
        return str

    def get_grid(self, goal_space = None):
        # here goal space is given as a list of integer values as done in the arguments. This it is converted into coordinates as done int \ours\grid\utils.py
        SUB_G = np.zeros((self.size, self.size))
        if goal_space != None:
            g_s_x = []
            g_s_y = []

            for i in goal_space:
                g_s_x.append((i) % self.size)
                g_s_y.append((i) // self.size)

            for j in range(len(goal_space)):
                SUB_G[g_s_y[j]][g_s_x[j]] = 1

        my = self.maze.shape[1]
        mx = self.maze.shape[2]

        maze_grid = [[0 for _ in range(mx)] for __ in range(my)]

        str = ''
        for y in range(my):
            for x in range(mx):
                if self.maze[BLOCK][y][x] == 1:
                    maze_grid[y][x] = 0
                    str += '  #'
                elif self.maze[AGENT][y][x] == 1:
                    str += '  A'
                    maze_grid[y][x] = 1
                elif self.maze[GOAL][y][x] == 1:
                    str += '  G'
                    maze_grid[y][x] = 2
                elif self.maze[PIT][y][x] == 1:
                    str += '  x'
                    maze_grid[y][x] = 3
                elif self.maze[KEY][y][x] == 1:
                    str += '  K'
                    maze_grid[y][x] = 4
                elif goal_space != None:
                    if SUB_G[y][x] == 1:
                        str += '  S'
                        maze_grid[y][x] = 5
                    else:
                        str += '   '
                        maze_grid[y][x] = 6
                else:
                    str += '   '
                    maze_grid[y][x] = 6
            str += '\n'

        return maze_grid


    def is_reachable(self, y, x):

        # if there is no block
        return self.maze[BLOCK][y][x] == 0

    def move_agent(self, direction):
        """
        part of forward model responsible for moving
        """

        #print("before:", self.agent_pos, self.maze[self.agent_pos[0]][self.agent_pos[1]][AGENT])

        y = self.agent_pos[0] + self.dy[direction]
        x = self.agent_pos[1] + self.dx[direction]

        if not self.is_reachable(y, x):
            return False

        # else move the agent

        self.maze[AGENT][self.agent_pos[0]][self.agent_pos[1]] = 0.0

        self.maze[AGENT][y][x] = 1.0
        self.agent_pos = [y, x]

        # moved the agent
        return True

    def step(self, action):
        if type(action) == np.ndarray:
            action = action.item()

        assert self.action_space.contains(action)
        # assert self.done is False

        constraint = 0
        info = {}

        # increment
        self.t += 1

        # for stochasticity, overwrite random action
        if self.random_action_prob > 0 and np.random.rand() < self.random_action_prob:
            action = np.random.choice(range(len(DX)))

        # move the agent
        moved = self.move_agent(action)

        # default reward
        reward = self.per_step_penalty

        #if the key is picked

        if self.maze[KEY][self.agent_pos[0]][self.agent_pos[1]] == 1.0:
            self.key_picked = True
        # if reached GOA                                                     L
        if self.maze[GOAL][self.agent_pos[0]][self.agent_pos[1]] == 1.0:
            if self.key_picked:
                reward = self.goal_reward*5
            else:
                reward = self.goal_reward
            self.done = True
        # if reached PIT
        if self.maze[PIT][self.agent_pos[0]][self.agent_pos[1]] == 1.0:
            constraint = self.constraint_cost


        # if max time steps reached
        if self.t >= self.max_step:
            self.done = True


        if self.t == 1:
            info['begin'] = True
        else:
            info['begin'] = False

        info['pit'] = constraint


        return self.observation(), reward, self.done, info




if __name__ == "__main__":

    env = PitWorld(size=14, one_hot_features=True, rand_goal=False)

    s = env.reset()

    print(env.maze.shape)
    print(env.maze[AGENT].shape)

    print("shape of obs:", s.shape)
    print("shape of obs:", env.reset().shape)
    print(env.to_string())
    print(env.agent_pos)

    # for a in range(4):
    for _ in range(50):

        print("0->u, 1->r, 2->d, 3->l")
        a = int(input())
        if a not in range(4):
            # go down
            if a == -1:
                s = env.reset()

            a = 0


        # print( DY[a], DX[a])
        s, r, d, info = env.step(a)

        # print(s)
        # print(r)
        print(env.to_string())
        print(env.agent_pos)
        print(env.t)
        print(env.done)
        print(info)
        # print(env.observation()[:,:,AGENT])
        # print(d)

    print("--------------------------------------------")
    s = env.reset()
    # print(env.to_string())
    print("--------------------------------------------")
