import matplotlib.pyplot as plt
import torch
import numpy as np

exp = "3"
gamma = 0.99

f_q_value_l = torch.load("f_q_value" + exp)
r_value_l = torch.load("r_value" + exp)


cost_l = torch.load("cost_"+ exp)

cost_current = []

for i in range(len(cost_l)):
    T = []
    prev_total = 0
    for k in range(len(cost_l[i])):
        prev_total = cost_l[i][k]
        T.append(cost_l[i][k] - prev_total)
    cost_current.append(T)

#print(cost_l)
C = []
C_R = []
Q = []
V_R = []

def compute_return(list, start):
    R = 0
    for i in range(start, len(list)):
        R = list[i] + gamma*R
    return R

def compute_return_reverse(list, start):
    R = 0
    for i in range(start+1):
        R = list[start-i] + gamma*R
    return R




forward_error = []
backward_error = []


for i in range(len(cost_current)):
    T = []
    f_e = 0
    r_e = 0
    eps_len = 0

    for k in range(len(cost_current[i])):
        eps_len +=1
        f_c = compute_return(cost_current[i], k)
        r_c = compute_return_reverse(cost_current[i], k)
        q = f_q_value_l[i][k]
        r_v = r_value_l[i][k]

        f_e += q - f_c
        r_e += r_v - r_c

    forward_error.append(f_e/eps_len)
    backward_error.append(r_e/eps_len)

r = torch.load("r" + exp)
c = torch.load("c" + exp)


figure, axis = plt.subplots(4)
axis[0].set_title("Non-HRL, BVF based Safe SARSA")
axis[2].plot(r, "b")
axis[3].plot(c, "r")
axis[3].set_ylim([0, 100])
axis[0].plot(forward_error, "g")
axis[0].set_ylim([0, 200])
axis[1].plot(backward_error, "y")
axis[1].set_ylim([0, 200])
#axis[3].plot(Ratio, "y")
axis[2].legend(["Reward (Evaluation)"])
axis[3].legend(["Cost ( Evaluation"])
axis[0].legend(["Forward Value Function Error"])
axis[1].legend(["Backward Value Function Error"])


plt.savefig("analysis")



"""          
plt.plot(Q)
plt.plot(C)
plt.show()
plt.close()

plt.plot(V_R)
plt.plot(C_R)
plt.show()
"""