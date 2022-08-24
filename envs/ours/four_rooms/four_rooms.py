import numpy as np
from gym import core, spaces
from gym.envs.registration import register

#adapted from the environemnt at https://github.com/arushijain94/SafeOptionCritic
"""
wwwwwwwwwwwww
w f   w   f w
w     w f   w
w  f  f  f  w
w     w     w
w f   w  f  w
ww wwww f   w
w f   www www
w     w fff w
wf    wfffffw
w     fff   w
w f   w  f  w
wwwwwwwwwwwww
"""
class Fourrooms(core.Env):
    def __init__(self, max_steps=200):

        layout = """\
wwwwwwwwwwwww
w f   w   f w
w     w f   w
w  f     f  w
w f   w     w
w     w  f  w
ww wwww f   w
w f   www www
w     w fff w
wf    wfffffw
w      ff   w
w f   w  f  w
wwwwwwwwwwwww
"""

        """
        Direction:
        0:U
        1:D
        2:L
        3:R
        Deterministic Actions
        Introducing variable rewards in "frozen"/ "slippery" state in range U[-15, 15] where expected value is zero as another states
        Reward for Goal state : 50
        Reward for Normal state/ hits wall:0
        """
        self.layout = layout

        self.occupancy = np.array([list(map(lambda c: 1 if c == 'w' else 0, line)) for line in layout.splitlines()])
        self.frozen = np.array([list(map(lambda c: 1 if c == 'f' else 0, line)) for line in layout.splitlines()])


        # From any state the agent can perform one of four actions, up, down, left or right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(np.sum(self.occupancy == 0))


        self.directions = [np.array((-1, 0)), np.array((1, 0)), np.array((0, -1)), np.array((0, 1))]
        self.rng = np.random.RandomState(1234)

        self.tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i, j] == 0:
                    self.tostate[(i, j)] = statenum
                    statenum += 1


        self.tocell = {v: k for k, v in self.tostate.items()}
        self.goal = 55
        self.size = statenum
        #self.init_states = list(range(self.observation_space.n))
        #self.init_states.remove(self.goal)
        self.init_states = [88]


        self.t = 0
        self.max_steps = max_steps

        self.reset()

    def empty_around(self, cell):
        avail = []
        for action in range(self.action_space.n):
            nextcell = tuple(cell + self.directions[action])
            if not self.occupancy[nextcell]:
                avail.append(nextcell)
        return avail

    def reset(self):

        self.t = 0
        state = self.rng.choice(self.init_states)
        self.currentcell = self.tocell[state]

        state_vec = np.zeros(shape=(self.size,))
        state_vec[state] = 1
        return state_vec

    def conver_state_num_state(self, num):
        state = num
        state_vec = np.zeros(shape=(self.size,))
        state_vec[state] = 1
        return state_vec

    def step(self, action):
        self.t += 1

        """
        The agent can perform one of four actions,
        up, down, left or right, which have a stochastic effect. With probability 2/3, the actions
        cause the agent to move one cell in the corresponding direction, and with probability 1/5,
        the agent moves instead in one of the other three directions, each with 1/15 probability. In
        either case, if the movement would take the agent into a wall then the agent remains in the
        same cell.
        """

        action = action[0]
        #action = action

        reward = 0
        if self.rng.uniform() < 1 / 5.:
            empty_cells = self.empty_around(self.currentcell)
            nextcell = empty_cells[self.rng.randint(len(empty_cells))]
        else:
            nextcell = tuple(self.currentcell + self.directions[action])

        if not self.occupancy[nextcell]:
            self.currentcell = nextcell

        state = self.tostate[self.currentcell]
        state_vec = np.zeros(shape=(self.size,))
        state_vec[state] = 1

        cost = 0
        reward = -1

        if self.frozen[self.currentcell]:
            cost = 10
            #print(10, self.currentcell)
        elif state == self.goal:
            reward = 1000

        done = state == self.goal

        info = {}
        info['pit'] = cost

        # if max time steps reached

        if self.t >= self.max_steps:
            done = True

        if self.t == 1:
            info['begin'] = True
        else:
            info['begin'] = False

        return state_vec, reward, done, info


env = Fourrooms()
print(np.shape(env.reset()))

O = np.array([list(map(lambda c: 1 if c == 'w' else 0, line)) for line in env.layout.splitlines()])
print(len(O), len(O[0]))

N = 55 #goal
k = 0
i = 0

while i != N + 1:

    if O[k // len(O)][k % len(O)] == 0:
        i += 1
        if i == N:
            O[k // len(O)][k % len(O)] = 10
    k += 1

N = 88 #start
k = 0
i = 0

while i != N + 1:

    if O[k // len(O)][k % len(O)] == 0:
        i += 1
        if i == N:
            O[k // len(O)][k % len(O)] = 20
    k += 1

for N in [52, 94, 26, 10]:

    i = 0
    k = 0
    while i != N+1:

        if O[k//len(O)][k%len(O)] == 0:
            i += 1
            if i == N:
                O[k // len(O)][k % len(O)] = 30
        k += 1

#print(O)