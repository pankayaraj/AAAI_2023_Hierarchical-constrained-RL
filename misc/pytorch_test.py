import torch
import numpy as np

size = 6
A = np.zeros((3, size, size))

A[1][5][5] =1.0

A[2][2][2] =1.0
A[2][4][2] =1.0
A[2][3][1] =1.0
A[2][5][4] =1.0
A[2][4][3] =1.0


print(A)
print((1000)*False)
B = np.reshape(A, (-1, size, size))
B = A[1].flatten()
print(B)