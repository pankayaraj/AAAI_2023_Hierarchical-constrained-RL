import torch
import numpy as np
import matplotlib.pyplot as plt

no_eps = 300000
l = no_eps//1000
a = []

no_experiments = 6

R_k = []
C_k = []

R_60 = []
C_60 = []
R_90 = []
C_90 = []

R_HRL_60 = []
C_HRL_60 = []
R_HRL_90 = []
C_HRL_90 = []


R_HRL_G_60 = []
C_HRL_G_60 = []
R_HRL_G_90 = []
C_HRL_G_90 = []


for i in range(no_experiments):
    R_60.append(torch.load("results/grid/safe_sarsa_key/60_new/r" + str(i + 1))[:l])
    C_60.append(torch.load("results/grid/safe_sarsa_key/60_new/c" + str(i + 1))[0:l])

    R_90.append(torch.load("results/grid/safe_sarsa_key/90_new/r" + str(i + 1))[:l])
    C_90.append(torch.load("results/grid/safe_sarsa_key/90_new/c" + str(i + 1))[0:l])

    #R_HRL_60.append(torch.load("results/grid/safe_lower_hrl_sarsa_key/c_20/r" + str(i + 1))[:l])
    #C_HRL_60.append(torch.load("results/grid/safe_lower_hrl_sarsa_key/c_20/c" + str(i + 1))[0:l])

    R_HRL_90.append(torch.load("results/grid/safe_cost_alloc_lower_hrl_sarsa_key/r" + str(i + 1))[:l])
    C_HRL_90.append(torch.load("results/grid/safe_cost_alloc_lower_hrl_sarsa_key/c" + str(i + 1))[0:l])

    R_HRL_G_60.append(torch.load("results/grid/safe_global_hrl_sarsa_key/60_new/r" + str(i + 1))[:l])
    C_HRL_G_60.append(torch.load("results/grid/safe_global_hrl_sarsa_key/60_new/c" + str(i + 1))[0:l])

    R_HRL_G_90.append(torch.load("results/grid/safe_global_hrl_sarsa_key/r" + str(i + 1))[:l])
    C_HRL_G_90.append(torch.load("results/grid/safe_global_hrl_sarsa_key/c" + str(i + 1))[0:l])

    #R_k.append(torch.load("results/grid/hrl_sarsa_key/r" + str(i + 1))[:l])
    #C_k.append(torch.load("results/grid/hrl_sarsa_key/c" + str(i + 1))[0:l])




R_avg_60 = []
C_avg_60 = []


R_avg_90 = []
C_avg_90 = []

#R_avg_HRL_60 = []
#C_avg_HRL_60 = []

R_avg_HRL_90 = []
C_avg_HRL_90 = []

R_avg_HRL_G_60 = []
C_avg_HRL_G_60 = []

R_avg_HRL_G_90 = []
C_avg_HRL_G_90 = []

#R_k_avg = []
#C_k_avg = []

for j in range(len(R_60)):


    r_avg_90 = []
    c_avg_90 = []

    r_avg_60 = []
    c_avg_60 = []

    #r_avg_HRL_60 = []
    #c_avg_HRL_60 = []

    r_avg_HRL_90 = []
    c_avg_HRL_90 = []

    r_avg_HRL_G_60 = []
    c_avg_HRL_G_60 = []

    r_avg_HRL_G_90 = []
    c_avg_HRL_G_90 = []

    #r_k_avg = []
    #c_k_avg = []

    for i in range(10,len(R_60[0])):
        r_avg_60.append(np.mean(R_60[j][i - 10:i]))
        c_avg_60.append(np.mean(C_60[j][i - 10:i]))

        r_avg_90.append(np.mean(R_90[j][i - 10:i]))
        c_avg_90.append(np.mean(C_90[j][i - 10:i]))

        #r_avg_HRL_60.append(np.mean(R_HRL_60[j][i - 10:i]))
        #c_avg_HRL_60.append(np.mean(C_HRL_60[j][i - 10:i]))

        r_avg_HRL_90.append(np.mean(R_HRL_90[j][i - 10:i]))
        c_avg_HRL_90.append(np.mean(C_HRL_90[j][i - 10:i]))

        r_avg_HRL_G_60.append(np.mean(R_HRL_G_60[j][i - 10:i]))
        c_avg_HRL_G_60.append(np.mean(C_HRL_G_60[j][i - 10:i]))

        r_avg_HRL_G_90.append(np.mean(R_HRL_G_90[j][i - 10:i]))
        c_avg_HRL_G_90.append(np.mean(C_HRL_G_90[j][i - 10:i]))

        #r_k_avg.append(np.mean(R_k[j][i - 10:i]))
        #c_k_avg.append(np.mean(C_k[j][i - 10:i]))


    R_avg_60.append(r_avg_60)
    C_avg_60.append(c_avg_60)

    R_avg_90.append(r_avg_90)
    C_avg_90.append(c_avg_90)

    #R_avg_HRL_60.append(r_avg_HRL_60)
    #C_avg_HRL_60.append(c_avg_HRL_60)

    R_avg_HRL_90.append(r_avg_HRL_90)
    C_avg_HRL_90.append(c_avg_HRL_90)

    R_avg_HRL_G_60.append(r_avg_HRL_G_60)
    C_avg_HRL_G_60.append(c_avg_HRL_G_60)

    R_avg_HRL_G_90.append(r_avg_HRL_G_90)
    C_avg_HRL_G_90.append(c_avg_HRL_G_90)

    #R_k_avg.append(r_k_avg)
    #C_k_avg.append(c_k_avg)

R_avg_60 = np.array(R_avg_60)
C_avg_60 = np.array(C_avg_60)
R_avg_90 = np.array(R_avg_90)
C_avg_90 = np.array(C_avg_90)
#R_avg_HRL_60 = np.array(R_avg_HRL_60)
#C_avg_HRL_60 = np.array(C_avg_HRL_60)
R_avg_HRL_90 = np.array(R_avg_HRL_90)
C_avg_HRL_90 = np.array(C_avg_HRL_90)
R_avg_HRL_G_60 = np.array(R_avg_HRL_G_60)
C_avg_HRL_G_60 = np.array(C_avg_HRL_G_60)
R_avg_HRL_G_90 = np.array(R_avg_HRL_G_90)
C_avg_HRL_G_90 = np.array(C_avg_HRL_G_90)
#R_k_avg = np.array(R_k_avg)
#C_k_avg = np.array(C_k_avg)

R_mean_60 = np.mean(R_avg_60, axis=0)
C_mean_60 = np.mean(C_avg_60, axis=0)
R_mean_90 = np.mean(R_avg_90, axis=0)
C_mean_90 = np.mean(C_avg_90, axis=0)
#R_mean_HRL_60 = np.mean(R_avg_HRL_60, axis=0)
#C_mean_HRL_60 = np.mean(C_avg_HRL_60, axis=0)
R_mean_HRL_90 = np.mean(R_avg_HRL_90, axis=0)
C_mean_HRL_90 = np.mean(C_avg_HRL_90, axis=0)
R_mean_HRL_G_60 = np.mean(R_avg_HRL_G_60, axis=0)
C_mean_HRL_G_60 = np.mean(C_avg_HRL_G_60, axis=0)
R_mean_HRL_G_90 = np.mean(R_avg_HRL_G_90, axis=0)
C_mean_HRL_G_90 = np.mean(C_avg_HRL_G_90, axis=0)
#R_k_mean = np.mean(R_k_avg, axis=0)
#C_k_mean = np.mean(C_k_avg, axis=0)


R_std_60 = np.std(R_avg_60, axis=0)
C_std_60 = np.std(C_avg_60, axis=0)
R_std_90 = np.std(R_avg_90, axis=0)
C_std_90 = np.std(C_avg_90, axis=0)
#R_std_HRL_60 = np.std(R_avg_HRL_60, axis=0)
#C_std_HRL_60 = np.std(C_avg_HRL_60, axis=0)
R_std_HRL_90 = np.std(R_avg_HRL_90, axis=0)
C_std_HRL_90 = np.std(C_avg_HRL_90, axis=0)
R_std_HRL_G_60 = np.std(R_avg_HRL_G_60, axis=0)
C_std_HRL_G_60 = np.std(C_avg_HRL_G_60, axis=0)
R_std_HRL_G_90 = np.std(R_avg_HRL_G_90, axis=0)
C_std_HRL_G_90 = np.std(C_avg_HRL_G_90, axis=0)
#R_k_std = np.std(R_k_avg, axis=0)
#C_k_std = np.std(C_k_avg, axis=0)

x = [i for i in range(len(R_mean_60))]
#legend = ["bvf-safe-sarsa", "hrl-safe-l-c-alloc", "hrl-safe-global-bvf", "unsafe_hrl"]
legend = ["bvf-without_hrl", "global-bvf-with-hrl", "hrl-with-cost-allocation"]
print(len(R_mean_60), len(R_mean_HRL_G_60))


fig, ax = plt.subplots(1, 1, figsize=(20, 10))
plt.tick_params(axis='both', which='major', labelsize=30)

#plt.plot(x, R_k_mean, label=legend[3], linewidth=5)
#plt.fill_between(x, R_k_mean + R_k_std, R_k_mean - R_k_std, alpha = 0.1)

plt.plot(x, R_mean_60, label=legend[0], linewidth=5)
plt.fill_between(x, R_mean_60 + R_std_60, R_mean_60 - R_std_60, alpha = 0.1)

#plt.plot(x, R_mean_HRL_60, label=legend[1], linewidth=5)
#plt.fill_between(x, R_mean_HRL_60 + R_std_HRL_60, R_mean_HRL_60 - R_std_HRL_60, alpha = 0.1)

plt.plot(x, R_mean_HRL_G_60, label=legend[2], linewidth=5)
plt.fill_between(x, R_mean_HRL_G_60 + R_std_HRL_G_60, R_mean_HRL_G_60 - R_std_HRL_G_60, alpha = 0.1)

plt.legend( prop={'size':40}, loc='upper left')
plt.xlabel('No of episodes X1000', size=40)
plt.ylabel("Reward", size=40)
plt.title("Reward comparision with a constriant limit of 60", size=40)
name = "figures/grid/comp/Rewards_Key_60"
plt.savefig(name)
plt.close(fig)


fig, ax = plt.subplots(1, 1, figsize=(20, 10))
plt.tick_params(axis='both', which='major', labelsize=30)


#plt.plot(x, R_k_mean, label=legend[3], linewidth=5)
#plt.fill_between(x, R_k_mean + R_k_std, R_k_mean - R_k_std, alpha = 0.1)

plt.plot(x, R_mean_90, label=legend[0], linewidth=5)
plt.fill_between(x, R_mean_90 + R_std_90, R_mean_90 - R_std_90, alpha = 0.1)

plt.plot(x, R_mean_HRL_90, label=legend[2], linewidth=5)
plt.fill_between(x, R_mean_HRL_90 + R_std_HRL_90, R_mean_HRL_90 - R_std_HRL_90, alpha = 0.1)

plt.plot(x, R_mean_HRL_G_90, label=legend[1], linewidth=5)
plt.fill_between(x, R_mean_HRL_G_90 + R_std_HRL_G_90, R_mean_HRL_G_90 - R_std_HRL_G_90, alpha = 0.1)

plt.legend( prop={'size':40}, loc='upper left')
plt.xlabel('No of episodes X1000', size=40)
plt.ylabel("Reward", size=40)
plt.title("Reward comparision with a constriant limit of 90", size=40)
name = "figures/grid/comp/Rewards_Key_90"
plt.savefig(name)
plt.close(fig)

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


fig, ax = plt.subplots(1, 1, figsize=(20, 10))
plt.tick_params(axis='both', which='major', labelsize=30)

#plt.plot(C_k_mean, label=legend[3], linewidth=3)
#plt.fill_between(x, C_k_mean + C_k_std, C_k_mean - C_k_std, alpha = 0.3)

plt.plot(C_mean_90, label=legend[0], linewidth=3)
plt.fill_between(x, C_mean_90 + C_std_90, C_mean_90 - C_std_90, alpha = 0.3)

plt.plot(C_mean_HRL_90, label=legend[2], linewidth=3)
plt.fill_between(x, C_mean_HRL_90 + C_std_HRL_90, C_mean_HRL_90 - C_std_HRL_90, alpha = 0.3)

plt.plot(C_mean_HRL_G_90, label=legend[1], linewidth=3)
plt.fill_between(x, C_mean_HRL_G_90 + C_std_HRL_G_90, C_mean_HRL_G_90 - C_std_HRL_G_90, alpha = 0.3)


plt.axhline(y=90, color='r', linestyle='--')
plt.legend(prop={'size':40}, loc='upper left')
plt.xlabel('No of episodes X1000', size=40)
plt.ylabel("Constraints", size=40)
plt.title("Constraint comparision with a constriant limit of 90", size=40)
name = "figures/grid/comp/Constraints_Key_90"

ax.set_ylim(0, 350)
plt.savefig(name)
plt.close(fig)
