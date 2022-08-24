import torch
import numpy as np
import matplotlib.pyplot as plt

no_eps = 210000
l = no_eps // 1000
l1 = 200
a = []

no_experiments = 3
n = 6

R_60 = [[] for _ in range(n)]
C_60 = [[] for _ in range(n)]

""""""



for i in range(no_experiments):


    R_60[0].append(torch.load("l_0_0_5/r" + str(i + 1))[:l])
    C_60[0].append(torch.load("l_0_0_5/c" + str(i + 1))[0:l])

    R_60[1].append(torch.load("l_0_1/r" + str(i + 1))[:l])
    C_60[1].append(torch.load("l_0_1/c" + str(i + 1))[0:l])

    R_60[2].append(torch.load("l_0_3/r" + str(i + 1))[:l])
    C_60[2].append(torch.load("l_0_3/c" + str(i + 1))[0:l])

    R_60[3].append(torch.load("l_0_5/r" + str(i + 1))[:l])
    C_60[3].append(torch.load("l_0_5/c" + str(i + 1))[0:l])

    R_60[4].append(torch.load("l_0_7/r" + str(i + 1))[:l])
    C_60[4].append(torch.load("l_0_7/c" + str(i + 1))[0:l])

    R_60[5].append(torch.load("l_1/r" + str(i + 1))[:l])
    C_60[5].append(torch.load("l_1/c" + str(i + 1))[0:l])


R_avg_60 = [[] for _ in range(n)]
C_avg_60 = [[] for _ in range(n)]

R_mean_60 = [[] for _ in range(n)]
C_mean_60 = [[] for _ in range(n)]
R_std_60  = [[] for _ in range(n)]
C_std_60  = [[] for _ in range(n)]

for j in range(len(R_60[0])):

    r_avg_60 = [[] for _ in range(n)]
    c_avg_60 = [[] for _ in range(n)]

    for k in range(n):
        for i in range(10, len(R_60[0][0])):

            r_avg_60[k].append(np.mean(R_60[k][j][i - 10:i]))
            c_avg_60[k].append(np.mean(C_60[k][j][i - 10:i]))


        R_avg_60[k].append(r_avg_60[k])
        C_avg_60[k].append(c_avg_60[k])

for k in range(n):
    R_avg_60[k] = np.array(R_avg_60[k])
    C_avg_60[k] = np.array(C_avg_60[k])

    R_mean_60[k] = np.mean(R_avg_60[k], axis=0)
    C_mean_60[k] = np.mean(C_avg_60[k], axis=0)

    R_std_60[k] = np.std(R_avg_60[k], axis=0)
    C_std_60[k] = np.std(C_avg_60[k], axis=0)

x = [i for i in range(len(R_mean_60[k]))]




legend = ["lam = 0.05", "lam = 0.1", "lam = 0.3", "lam = 0.5", "lam = 0.7", "lam = 1", ]

#LOW LAGRANGIAN
fig, ax = plt.subplots(1, 1, figsize=(55, 35))
plt.tick_params(axis='both', which='major', labelsize=30)

for k in range(n):
    plt.plot(x, R_mean_60[k], label=legend[k], linewidth=10, )
    plt.fill_between(x, R_mean_60[k] + R_std_60[k], R_mean_60[k] - R_std_60[k], alpha=0.1,)


plt.legend( prop={'size':48},loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=6)
plt.xlabel('No of episodes X1000', size=80)
plt.ylabel("Reward", size=80)
plt.title("Reward comparision with a constriant limit of 60", size=80)
name = "Rewards_low_lagrangian"
plt.savefig(name)
plt.close(fig)
"""

fig, ax = plt.subplots(1, 1, figsize=(20, 10))
plt.tick_params(axis='both', which='major', labelsize=30)

plt.plot(C_mean_HRL_60, label=legend[0], linewidth=3, color='r')
plt.fill_between(x, C_mean_HRL_60 + C_std_HRL_60, C_mean_HRL_60 - C_std_HRL_60, alpha=0.3, color='r')

plt.plot(C_mean_HRL_90, label=legend[1], linewidth=3, color='g')
plt.fill_between(x, C_mean_HRL_90 + C_std_HRL_90, C_mean_HRL_90 - C_std_HRL_90, alpha=0.3, color='g')

plt.axhline(y=90, color='r', linestyle='--')
plt.axhline(y=60, color='g', linestyle='--')
plt.legend(prop={'size': 40}, loc='upper left')
plt.xlabel('No of episodes X1000', size=40)
plt.ylabel("Constraints", size=40)
plt.title("Constraint comparision with a constriant limit of 90", size=40)
name = "figures/grid/individual_comp/Constraints_low_lagrangian"

ax.set_ylim(0, 350)
plt.savefig(name)
plt.close(fig)


#BVF

fig, ax = plt.subplots(1, 1, figsize=(20, 10))
plt.tick_params(axis='both', which='major', labelsize=30)

plt.plot(x, R_mean_60, label=legend[0], linewidth=5, color='r')
plt.fill_between(x, R_mean_60 + R_std_HRL_60, R_mean_60 - R_std_60, alpha=0.1, color='r')

plt.plot(x, R_mean_90, label=legend[1], linewidth=5, color='g')
plt.fill_between(x, R_mean_90 + R_std_90, R_mean_90 - R_std_90, alpha=0.1, color='g')

plt.legend(prop={'size': 20}, loc='upper left')
plt.xlabel('No of episodes X1000', size=40)
plt.ylabel("Reward", size=40)
plt.title("Reward comparision with a constriant limit of 90", size=40)
name = "figures/grid/individual_comp/Rewards_bvf"
plt.savefig(name)
plt.close(fig)


fig, ax = plt.subplots(1, 1, figsize=(20, 10))
plt.tick_params(axis='both', which='major', labelsize=30)

plt.plot(C_mean_60, label=legend[0], linewidth=3, color='r')
plt.fill_between(x, C_mean_60 + C_std_60, C_mean_60 - C_std_60, alpha=0.3, color='r')

plt.plot(C_mean_90, label=legend[1], linewidth=3, color='g')
plt.fill_between(x, C_mean_90 + C_std_90, C_mean_90 - C_std_90, alpha=0.3, color='g')

plt.axhline(y=90, color='r', linestyle='--')
plt.axhline(y=60, color='g', linestyle='--')
plt.legend(prop={'size': 20}, loc='upper left')
plt.xlabel('No of episodes X1000', size=40)
plt.ylabel("Constraints", size=40)
plt.title("Constraint comparision with a constriant limit of 90", size=40)
name = "figures/grid/individual_comp/Constraints_bvf"

ax.set_ylim(0, 350)
plt.savefig(name)
plt.close(fig)


"""
"""
fig, ax = plt.subplots(1, 1, figsize=(20, 10))
plt.tick_params(axis='both', which='major', labelsize=30)

#plt.plot(C_k_mean, label=legend[3], linewidth=3)
#plt.fill_between(x, C_k_mean + C_k_std, C_k_mean - C_k_std, alpha = 0.3)

plt.plot(C_mean_60, label=legend[0], linewidth=3)
plt.fill_between(x, C_mean_60 + C_std_60, C_mean_60 - C_std_60, alpha = 0.3)

#plt.plot(C_mean_HRL_60, label=legend[1], linewidth=3)
#plt.fill_between(x, C_mean_HRL_60 + C_std_HRL_60, C_mean_HRL_60 - C_std_HRL_60, alpha = 0.3)

plt.plot(C_mean_HRL_G_60, label=legend[2], linewidth=3)
plt.fill_between(x, C_mean_HRL_G_60 + C_std_HRL_G_60, C_mean_HRL_G_60 - C_std_HRL_G_60, alpha = 0.3)

plt.axhline(y=60, color='r', linestyle='--')
plt.legend(prop={'size':40}, loc='upper left')
plt.xlabel('No of episodes X1000', size=40)
plt.ylabel("Constraints", size=40)
plt.title("Constraint comparision with a constriant limit of 60", size=40)
name = "figures/grid/comp/Constraints_Key_60"

ax.set_ylim(0, 350)
plt.savefig(name)
plt.close(fig)
"""
