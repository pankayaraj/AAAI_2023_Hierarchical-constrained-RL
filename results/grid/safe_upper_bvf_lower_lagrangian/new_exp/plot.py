import matplotlib.pyplot as plt
import torch
import numpy as np

exp = "1"
gamma = 0.99

f_q_value_l = torch.load("f_q_value_u" + exp)
r_value_l = torch.load("r_value_u" + exp)


value_g1 = torch.load("value_g1_" + exp)
value_g2 = torch.load("value_g2_" + exp)
value_g3 = torch.load("value_g3_" + exp)

print(len(value_g1), len(value_g1[0]))

v1 = []
v2 = []
v3 = []


for i in range(len(value_g1)):
    v1.append(value_g1[i][0])

for i in range(len(value_g2)):
    v2.append(value_g2[i][0])

for i in range(len(value_g3)):
    v3.append(value_g3[i][0])


F_Q = []

for i in range(len(f_q_value_l)):
    for j in range(len(f_q_value_l[i])):
        F_Q.append(f_q_value_l[i][j])

B_V = []

for i in range(len(r_value_l)):
    for j in range(len(r_value_l[i])):
        B_V.append(r_value_l[i][j])


cost_l = torch.load("cost_U_"+ exp)
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



figure, axis = plt.subplots(3)
axis[0].set_title("HRL Safe SARSA where cost allocation for lower level is treated as an action")
axis[1].plot(r, "b")
axis[2].plot(c, "r")
axis[2].set_ylim([0, 100])
axis[0].plot(v1, "b")
#axis[0].set_ylim([0, 200])
axis[0].plot(v2, "g")
axis[0].plot(v3, "y")

#axis[3].plot(F_Q[-10000:-1])
#axis[4].plot(B_V[-10000:-1])
#axis[1].set_ylim([0, 200])
#axis[3].plot(Ratio, "y")
axis[1].legend(["Reward (Evaluation)"])
axis[2].legend(["Cost ( Evaluation"])
#axis[0].legend(["Random Goal"])
axis[0].legend(["Random Goal", "Key", "Goal"])

plt.show()



"""          
plt.plot(Q)
plt.plot(C)
plt.show()
plt.close()

plt.plot(V_R)
plt.plot(C_R)
plt.show()
"""