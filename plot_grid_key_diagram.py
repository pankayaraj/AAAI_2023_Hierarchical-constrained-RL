import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from envs.ours.gird.safety_gridworld_with_key import PitWorld_Key


goal_space = [212, 86, 160, 163, 282, 135, 200]

env = PitWorld_Key(size=18,
                           max_step=200,
                           per_step_penalty=-1.0,
                           goal_reward=1000.0,
                           obstace_density=0.3,
                           constraint_cost=10.0,
                           random_action_prob=0.005,
                           one_hot_features=True,
                           rand_goal=False,  # for testing purposes
                           )


data = env.get_grid(goal_space=goal_space)


cmap = colors.ListedColormap(['black', 'blue', "green", "red", "aqua", "yellow", "white"])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
norm = colors.BoundaryNorm(bounds, cmap.N)


fig, ax = plt.subplots()
ax.imshow(data, cmap=cmap, norm=norm)

# draw gridlines
ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2, )
ax.legend=['black', 'blue', "green", "red", "aqua", "yellow", "white"]
ax.set_xticks(np.arange(-0.5, 18.5, 1))
ax.set_yticks(np.arange(-0.5, 18.5, 1))

plt.show()
