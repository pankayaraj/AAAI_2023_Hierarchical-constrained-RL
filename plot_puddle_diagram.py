import numpy

from envs.ours.four_rooms.four_rooms import Fourrooms
import copy

array = [[0 for i in range(10)] for j in range(10)]
array[0][0] = 1
array[1][9] = 2

cost1 = [[7, 3], [0, 6]]
cost2 = [[10, 0], [8.5, 5]]
cost3 = [[6, 9], [5, 10]]

for i in range(10):
    for j in range(10):
        if (i <= cost1[0][0] and i >= cost1[1][0] and j >= cost1[0][1] and j <= cost1[1][1]):
            print(i,j)
            array[i][j] = 3
        else:
            pass

        if (i <= cost2[0][0] and i >= cost2[1][0] and j >= cost2[0][1] and j <= cost2[1][1]):
            print(i,j)
            array[i][j] = 3
        else:
            pass


        if (i <= cost3[0][0] and i >= cost3[1][0] and j >= cost3[0][1] and j <= cost3[1][1]):
            print(i,j)
            array[i][j] = 3
        else:
            pass


#sub goal

array[7][1] = 4
array[9][9] = 4
array[5][8] = 4


import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

""""""
data = array

# create discrete colormap
cmap = colors.ListedColormap(['white','blue', "green", "red", "yellow"])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
norm = colors.BoundaryNorm(bounds, cmap.N)


fig, ax = plt.subplots()
ax.imshow(data, cmap=cmap, norm=norm)
ax.axis('off')
# draw gridlines
ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2, b=None)
#ax.legend=['red', 'white', "black", "green", "blue", "yellow"]
ax.set_xticks(np.arange(-0.5, 9.95, 1))
ax.set_yticks(np.arange(-0.5, 9.95, 1))

plt.show()
