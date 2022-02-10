from common.ours.grid.utils import Goal_Space
import numpy as np
G = Goal_Space([], 14)


A  = []
for i in range(14): #y
    a = []
    for j in range(14): #x
        a.append(G.convert_cooridnates_to_value(j, i))
    A.append(a)


start = 180

goal = [124, 83, 76, 8, 118, 57]

print(G.convert_value_to_coordinates(143))
A = np.array(A)
print(A)