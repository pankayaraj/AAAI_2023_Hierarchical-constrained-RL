from gym_minigrid.wrappers import *
env = gym.make("MiniGrid-DoorKey-16x16-v0")
env = OneHotPartialObsWrapper(env, tile_size=16)
print(env.reset())