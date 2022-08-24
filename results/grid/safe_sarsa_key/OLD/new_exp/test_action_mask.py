import torch
import matplotlib.pyplot as plt
import numpy as np


exp_no = "1"
r = torch.load("r" + str(exp_no))

#plt.plot(r)
#plt.show()


a_m = torch.load("action_mask" + str(exp_no))
eps = [len(x) for x in a_m]


T = 0
Ratio = []
True_list = []
False_list = []
Random = []

Ratio_Low = []
True_list_low = []
False_list_low = []
print(len(a_m), len(r))

for i in range(len(a_m)):
    true = 0
    false = 0
    random = 0
    for e in a_m[i]:
        true_low = 0
        false_low = 0

        if e != None:
            pass
            T += 1

            if e.item() == True:
                true += 1
                true_low += 1
            elif e.item() == False:
                false += 1
                false_low += 1
        else:
            random += 1
        True_list_low.append(true_low)
        False_list_low.append(false_low)
    True_list.append(true)
    False_list.append(false)
    Random.append(random)

for i in range(len(True_list)):
    if True_list[i] != 0:
        Ratio.append(False_list[i]/True_list[i])
    else:
        Ratio.append(0)


figure, axis = plt.subplots(4)
axis[0].plot(r, "b")
axis[1].plot(True_list, "r")
axis[1].set_ylim([0, 20])
axis[2].plot(False_list, "g")
axis[2].set_ylim([0, 30])
axis[3].plot(Random, "y")
axis[3].set_ylim([0, 30])
#axis[3].plot(Ratio, "y")
axis[0].legend(["Reward (Evaluation)"])
axis[1].legend(["Minimizes the cost"])
axis[2].legend(["Do not minimize the cost"])
axis[3].legend(["Random Action"])
#axis[3].legend(["Ratio"])


plt.show()

#print(T)
#c_e = torch.load("cost_estimate_2")
#print(c_e[280])
#c = torch.load("cost_2")
#print(c[280])