import torch
import numpy as np
import matplotlib.pyplot as plt

no_eps = 500000
l = no_eps//1000
a = []

no_experiments = 3


names = ["l_0_0_5", "l_0_1", "l_0_3", "l_0_5"]

R_90 = []
C_90 = []


for j in range(len(names)):
    T_R = []
    T_C = []
    for i in range(no_experiments):

        T_R.append(torch.load("results/grid/safe_upper_bvf_lower_lagrangian/" + str(names[j]) + "/r" + str(i + 1))[:l])
        T_C.append(torch.load("results/grid/safe_upper_bvf_lower_lagrangian/" + str(names[j]) + "/c" + str(i + 1))[0:l])

    R_90.append(T_R)
    C_90.append(T_C)

R_avg_90 = []
C_avg_90 = []


for x in range(len(names)):
    T_R = []
    T_C = []

    for j in range(len(R_90[0])):


        r_avg_90 = []
        c_avg_90 = []



        for i in range(10,len(R_90[0][0])):

            r_avg_90.append(np.mean(R_90[x][j][i - 10:i]))
            c_avg_90.append(np.mean(C_90[x][j][i - 10:i]))

        T_R.append(r_avg_90)
        T_C.append(c_avg_90)

    R_avg_90.append(T_R)
    C_avg_90.append(T_C)


R_avg_90 = [np.array(R_avg_90[i]) for i in range(len(names))]
C_avg_90 = [np.array(C_avg_90[i]) for i in range(len(names))]

R_mean_90 = [np.mean(R_avg_90[i], axis=0) for i in range(len(names))]
C_mean_90 = [np.mean(C_avg_90[i], axis=0) for i in range(len(names))]


R_std_90 = [np.std(R_avg_90[i], axis=0) for i in range(len(names))]
C_std_90 = [np.std(C_avg_90[i], axis=0) for i in range(len(names))]



x = [i for i in range(len(R_mean_90[0]))]
#legend = ["bvf-safe-sarsa", "hrl-safe-l-c-alloc", "hrl-safe-global-bvf", "unsafe_hrl"]
legend = ["0.05", "0.1", "0.3", "0.5"]





fig, ax = plt.subplots(1, 1, figsize=(20, 10))
plt.tick_params(axis='both', which='major', labelsize=30)


#plt.plot(x, R_k_mean, label=legend[3], linewidth=5)
#plt.fill_between(x, R_k_mean + R_k_std, R_k_mean - R_k_std, alpha = 0.1)

plt.plot(x, R_mean_90[0], label=legend[0], linewidth=5)
plt.fill_between(x, R_mean_90[0] + R_std_90[0], R_mean_90[0] - R_std_90[0], alpha = 0.1)

plt.plot(x, R_mean_90[1], label=legend[1], linewidth=5)
plt.fill_between(x, R_mean_90[1] + R_std_90[1], R_mean_90[1] - R_std_90[0], alpha = 0.1)

plt.plot(x, R_mean_90[2], label=legend[2], linewidth=5)
plt.fill_between(x, R_mean_90[2] + R_std_90[2], R_mean_90[2] - R_std_90[2], alpha = 0.1)

plt.plot(x, R_mean_90[3], label=legend[3], linewidth=5)
plt.fill_between(x, R_mean_90[3] + R_std_90[3], R_mean_90[3] - R_std_90[3], alpha = 0.1)

plt.legend( prop={'size':40}, loc='upper left')
plt.xlabel('No of episodes X1000', size=40)
plt.ylabel("Reward", size=40)
plt.title("Reward comparision with a constriant limit of 90", size=40)
name = "figures/grid/comp/Rewards_Comparision"
plt.savefig(name)
plt.close(fig)




fig, ax = plt.subplots(1, 1, figsize=(20, 10))
plt.tick_params(axis='both', which='major', labelsize=30)

#plt.plot(C_k_mean, label=legend[3], linewidth=3)
#plt.fill_between(x, C_k_mean + C_k_std, C_k_mean - C_k_std, alpha = 0.3)

plt.plot(C_mean_90[0], label=legend[0], linewidth=3)
plt.fill_between(x, C_mean_90[0] + C_std_90[0], C_mean_90[0] - C_std_90[0], alpha = 0.3)

plt.plot(C_mean_90[1], label=legend[1], linewidth=3)
plt.fill_between(x, C_mean_90[1] + C_std_90[1], C_mean_90[1] - C_std_90[1], alpha = 0.3)

plt.plot(C_mean_90[2], label=legend[2], linewidth=3)
plt.fill_between(x, C_mean_90[2] + C_std_90[2], C_mean_90[2] - C_std_90[2], alpha = 0.3)

plt.plot(C_mean_90[0], label=legend[3], linewidth=3)
plt.fill_between(x, C_mean_90[3] + C_std_90[3], C_mean_90[3] - C_std_90[3], alpha = 0.3)

plt.axhline(y=90, color='r', linestyle='--')
plt.legend(prop={'size':40}, loc='upper left')
plt.xlabel('No of episodes X1000', size=40)
plt.ylabel("Constraints", size=40)
plt.title("Constraint comparision with a constriant limit of 90", size=40)
name = "figures/grid/comp/Constraints_Comparision"

ax.set_ylim(0, 350)
plt.savefig(name)
plt.close(fig)
