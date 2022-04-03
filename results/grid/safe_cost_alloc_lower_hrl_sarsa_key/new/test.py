import torch

a = torch.load("adj_2")

t = 0
for x in a[0:6000]:
    t += 1
    if x[0][4] == True:
        if x[0][1].item() == 212:
            print(x[0][2])
            print(x[0][0])
            print(t)