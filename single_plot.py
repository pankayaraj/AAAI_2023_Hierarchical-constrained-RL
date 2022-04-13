import torch
import numpy as np
import matplotlib.pyplot as plt

no_eps = 500000
l = no_eps//1000
a = []

no_experiments = 4

R_90 = []
C_90 = []



for i in range(no_experiments):

    R_90.append(torch.load("results/grid/safe_upper_bvf_lower_lagrangian/r" + str(i + 1))[:l])
    C_90.append(torch.load("results/grid/safe_upper_bvf_lower_lagrangian/c" + str(i + 1))[0:l])




R_avg_90 = []
C_avg_90 = []



for j in range(len(R_90)):


    r_avg_90 = []
    c_avg_90 = []



    for i in range(10,len(R_90[0])):


        r_avg_90.append(np.mean(R_90[j][i - 10:i]))
        c_avg_90.append(np.mean(C_90[j][i - 10:i]))



    R_avg_90.append(r_avg_90)
    C_avg_90.append(c_avg_90)


    #R_k_avg.append(r_k_avg)
    #C_k_avg.append(c_k_avg)

R_avg_90 = np.array(R_avg_90)
C_avg_90 = np.array(C_avg_90)

R_mean_90 = np.mean(R_avg_90, axis=0)
C_mean_90 = np.mean(C_avg_90, axis=0)
#R_mean_HRL_60 = np.mean(R_avg_HRL_60, axis=0)
#C_mean_HRL_60 = np.mean(C_avg_HRL_60, axis=0)
#R_k_mean = np.mean(R_k_avg, axis=0)
#C_k_mean = np.mean(C_k_avg, axis=0)


R_std_90 = np.std(R_avg_90, axis=0)
C_std_90 = np.std(C_avg_90, axis=0)
#R_std_HRL_60 = np.std(R_avg_HRL_60, axis=0)
#C_std_HRL_60 = np.std(C_avg_HRL_60, axis=0)


x = [i for i in range(len(R_mean_90))]
#legend = ["bvf-safe-sarsa", "hrl-safe-l-c-alloc", "hrl-safe-global-bvf", "unsafe_hrl"]
legend = [""]






fig, ax = plt.subplots(1, 1, figsize=(20, 10))
plt.tick_params(axis='both', which='major', labelsize=30)


#plt.plot(x, R_k_mean, label=legend[3], linewidth=5)
#plt.fill_between(x, R_k_mean + R_k_std, R_k_mean - R_k_std, alpha = 0.1)

plt.plot(x, R_mean_90, label=legend[0], linewidth=5)
plt.fill_between(x, R_mean_90 + R_std_90, R_mean_90 - R_std_90, alpha = 0.1)


plt.legend( prop={'size':40}, loc='upper left')
plt.xlabel('No of episodes X1000', size=40)
plt.ylabel("Reward", size=40)
plt.title("Reward comparision with a constriant limit of 90", size=40)
name = "figures/grid/comp/Rewards_Singular"
plt.savefig(name)
plt.close(fig)




fig, ax = plt.subplots(1, 1, figsize=(20, 10))
plt.tick_params(axis='both', which='major', labelsize=30)

#plt.plot(C_k_mean, label=legend[3], linewidth=3)
#plt.fill_between(x, C_k_mean + C_k_std, C_k_mean - C_k_std, alpha = 0.3)

plt.plot(C_mean_90, label=legend[0], linewidth=3)
plt.fill_between(x, C_mean_90 + C_std_90, C_mean_90 - C_std_90, alpha = 0.3)


plt.axhline(y=90, color='r', linestyle='--')
plt.legend(prop={'size':40}, loc='upper left')
plt.xlabel('No of episodes X1000', size=40)
plt.ylabel("Constraints", size=40)
plt.title("Constraint comparision with a constriant limit of 90", size=40)
name = "figures/grid/comp/Constraints_Singular"

ax.set_ylim(0, 350)
plt.savefig(name)
plt.close(fig)
