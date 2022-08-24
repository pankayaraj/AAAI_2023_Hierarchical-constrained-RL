import torch
import numpy as np
import matplotlib.pyplot as plt

no_eps = 230000
l = no_eps // 1000
l1 = 230
a = []

no_experiments = 6

R_k = []
C_k = []

R_k_HRL = []
C_k_HRL = []

R_60 = []
C_60 = []
R_90 = []
C_90 = []

R_lyp_60 = []
C_lyp_60 = []

R_HRL_60 = []
C_HRL_60 = []
R_HRL_90 = []
C_HRL_90 = []

R_HRL_G_60 = []
C_HRL_G_60 = []
R_HRL_G_90 = []
C_HRL_G_90 = []

for i in range(no_experiments):
    R_k.append(torch.load("results/four_rooms/sarsa/r" + str(i + 1))[:l1])  # non safe non HRL
    C_k.append(torch.load("results/four_rooms/sarsa/c" + str(i + 1))[0:l1])

    R_k_HRL.append(torch.load("results/four_rooms/hrl_sarsa/r" + str(i + 1))[:l1])  # non safe HRL
    C_k_HRL.append(torch.load("results/four_rooms/hrl_sarsa/c" + str(i + 1))[0:l1])

    R_60.append(torch.load("results/four_rooms/safe_sarsa/30/r" + str(i + 1))[:l])  # bvf sarsa
    C_60.append(torch.load("results/four_rooms/safe_sarsa/30/c" + str(i + 1))[0:l])

    R_HRL_60.append(torch.load("results/four_rooms/safe_upper_bvf_lower_lagrangian/30/r" + str(i + 1))[:l])  # lower level lagrangian
    C_HRL_60.append(torch.load("results/four_rooms/safe_upper_bvf_lower_lagrangian/30/c" + str(i + 1))[0:l])

    R_lyp_60.append(torch.load("results/four_rooms/lyp_sarsa/new_30/r" + str(i + 1))[:l])  # bvf sarsa
    C_lyp_60.append(torch.load("results/four_rooms/lyp_sarsa/new_30/c" + str(i + 1))[0:l])


    R_90.append(torch.load("results/grid/safe_sarsa_key/new_90/r" + str(i + 1))[:l])  # bvf sarsa
    C_90.append(torch.load("results/grid/safe_sarsa_key/new_90/c" + str(i + 1))[0:l])

    R_HRL_90.append(torch.load("results/grid/safe_upper_bvf_lower_lagrangian/new_90/r" + str(i + 1))[:l])  # lower level lagrangian
    C_HRL_90.append(torch.load("results/grid/safe_upper_bvf_lower_lagrangian/new_90/c" + str(i + 1))[0:l])

    # R_HRL_G_60.append(torch.load("results/grid/safe_global_hrl_sarsa_key/new_60/r" + str(i + 1))[:l])  #global bvf hrl
    # C_HRL_G_60.append(torch.load("results/grid/safe_global_hrl_sarsa_key/new_60/c" + str(i + 1))[0:l])

    R_HRL_G_90.append(torch.load("results/grid/safe_dual_global_hrl_sarsa_key/new_90/r" + str(i + 1))[:l])  # global bvf hrl
    C_HRL_G_90.append(torch.load("results/grid/safe_dual_global_hrl_sarsa_key/new_90/c" + str(i + 1))[0:l])




R_avg_60 = []
C_avg_60 = []

R_avg_90 = []
C_avg_90 = []

R_lyp_avg_60 = []
C_lyp_avg_60 = []

R_avg_HRL_60 = []
C_avg_HRL_60 = []

R_avg_HRL_90 = []
C_avg_HRL_90 = []

R_avg_HRL_G_60 = []
C_avg_HRL_G_60 = []

R_avg_HRL_G_90 = []
C_avg_HRL_G_90 = []

R_k_avg = []
C_k_avg = []

R_k_HRL_avg = []
C_k_HRL_avg = []

T = 10
for j in range(len(R_k)):
    r_k_avg = []
    c_k_avg = []
    r_k_hrl_avg = []
    c_k_hrl_avg = []

    for i in range(T, len(R_k[0])):
        r_k_avg.append(np.mean(R_k[j][i - T:i]))
        c_k_avg.append(np.mean(C_k[j][i - T:i]))

        r_k_hrl_avg.append(np.mean(R_k_HRL[j][i - T:i]))
        c_k_hrl_avg.append(np.mean(C_k_HRL[j][i - T:i]))

    R_k_avg.append(r_k_avg)
    C_k_avg.append(c_k_avg)

    R_k_HRL_avg.append(r_k_hrl_avg)
    C_k_HRL_avg.append(c_k_hrl_avg)

for j in range(len(R_90)):

    r_avg_90 = []
    c_avg_90 = []

    r_avg_60 = []
    c_avg_60 = []

    r_lyp_avg_60 = []
    c_lyp_avg_60 = []

    r_avg_HRL_60 = []
    c_avg_HRL_60 = []

    r_avg_HRL_90 = []
    c_avg_HRL_90 = []

    r_avg_HRL_G_60 = []
    c_avg_HRL_G_60 = []

    r_avg_HRL_G_90 = []
    c_avg_HRL_G_90 = []

    for i in range(T, len(R_90[0])):
        r_avg_60.append(np.mean(R_60[j][i - T:i]))
        c_avg_60.append(np.mean(C_60[j][i - T:i]))

        r_avg_90.append(np.mean(R_90[j][i - T:i]))
        c_avg_90.append(np.mean(C_90[j][i - T:i]))

        r_lyp_avg_60.append(np.mean(R_lyp_60[j][i - T:i]))
        c_lyp_avg_60.append(np.mean(C_lyp_60[j][i - T:i]))

        r_avg_HRL_60.append(np.mean(R_HRL_60[j][i - T:i]))
        c_avg_HRL_60.append(np.mean(C_HRL_60[j][i - T:i]))

        r_avg_HRL_90.append(np.mean(R_HRL_90[j][i - T:i]))
        c_avg_HRL_90.append(np.mean(C_HRL_90[j][i - T:i]))

        # r_avg_HRL_G_60.append(np.mean(R_HRL_G_60[j][i - 10:i]))
        # c_avg_HRL_G_60.append(np.mean(C_HRL_G_60[j][i - 10:i]))

        r_avg_HRL_G_90.append(np.mean(R_HRL_G_90[j][i - T:i]))
        c_avg_HRL_G_90.append(np.mean(C_HRL_G_90[j][i - T:i]))

    R_avg_60.append(r_avg_60)
    C_avg_60.append(c_avg_60)

    R_avg_90.append(r_avg_90)
    C_avg_90.append(c_avg_90)

    R_lyp_avg_60.append(r_lyp_avg_60)
    C_lyp_avg_60.append(c_lyp_avg_60)

    R_avg_HRL_60.append(r_avg_HRL_60)
    C_avg_HRL_60.append(c_avg_HRL_60)

    R_avg_HRL_90.append(r_avg_HRL_90)
    C_avg_HRL_90.append(c_avg_HRL_90)

    # R_avg_HRL_G_60.append(r_avg_HRL_G_60)
    # C_avg_HRL_G_60.append(c_avg_HRL_G_60)

    R_avg_HRL_G_90.append(r_avg_HRL_G_90)
    C_avg_HRL_G_90.append(c_avg_HRL_G_90)

R_avg_60 = np.array(R_avg_60)
C_avg_60 = np.array(C_avg_60)
R_avg_90 = np.array(R_avg_90)
C_avg_90 = np.array(C_avg_90)
R_lyp_avg_60 = np.array(R_lyp_avg_60)
C_lyp_avg_60 = np.array(C_lyp_avg_60)

R_avg_HRL_60 = np.array(R_avg_HRL_60)
C_avg_HRL_60 = np.array(C_avg_HRL_60)
R_avg_HRL_90 = np.array(R_avg_HRL_90)
C_avg_HRL_90 = np.array(C_avg_HRL_90)
# R_avg_HRL_G_60 = np.array(R_avg_HRL_G_60)
# C_avg_HRL_G_60 = np.array(C_avg_HRL_G_60)
R_avg_HRL_G_90 = np.array(R_avg_HRL_G_90)
C_avg_HRL_G_90 = np.array(C_avg_HRL_G_90)
R_k_avg = np.array(R_k_avg)
C_k_avg = np.array(C_k_avg)
R_k_HRL_avg = np.array(R_k_HRL_avg)
C_k_HRL_avg = np.array(C_k_HRL_avg)

R_mean_60 = np.mean(R_avg_60, axis=0)
C_mean_60 = np.mean(C_avg_60, axis=0)
R_mean_90 = np.mean(R_avg_90, axis=0)
C_mean_90 = np.mean(C_avg_90, axis=0)
R_lyp_mean_60 = np.mean(R_lyp_avg_60, axis=0)
C_lyp_mean_60 = np.mean(C_lyp_avg_60, axis=0)

R_mean_HRL_60 = np.mean(R_avg_HRL_60, axis=0)
C_mean_HRL_60 = np.mean(C_avg_HRL_60, axis=0)
R_mean_HRL_90 = np.mean(R_avg_HRL_90, axis=0)
C_mean_HRL_90 = np.mean(C_avg_HRL_90, axis=0)
# R_mean_HRL_G_60 = np.mean(R_avg_HRL_G_60, axis=0)
# C_mean_HRL_G_60 = np.mean(C_avg_HRL_G_60, axis=0)
R_mean_HRL_G_90 = np.mean(R_avg_HRL_G_90, axis=0)
C_mean_HRL_G_90 = np.mean(C_avg_HRL_G_90, axis=0)
R_k_mean = np.mean(R_k_avg, axis=0)
C_k_mean = np.mean(C_k_avg, axis=0)
R_k_HRL_mean = np.mean(R_k_HRL_avg, axis=0)
C_k_HRL_mean = np.mean(C_k_HRL_avg, axis=0)


R_std_60 = np.std(R_avg_60, axis=0)
C_std_60 = np.std(C_avg_60, axis=0)
R_std_90 = np.std(R_avg_90, axis=0)
C_std_90 = np.std(C_avg_90, axis=0)

R_lyp_std_60 = np.std(R_lyp_avg_60, axis=0)
C_lyp_std_60 = np.std(C_lyp_avg_60, axis=0)

R_std_HRL_60 = np.std(R_avg_HRL_60, axis=0)
C_std_HRL_60 = np.std(C_avg_HRL_60, axis=0)
R_std_HRL_90 = np.std(R_avg_HRL_90, axis=0)
C_std_HRL_90 = np.std(C_avg_HRL_90, axis=0)
# R_std_HRL_G_60 = np.std(R_avg_HRL_G_60, axis=0)
# C_std_HRL_G_60 = np.std(C_avg_HRL_G_60, axis=0)
R_std_HRL_G_90 = np.std(R_avg_HRL_G_90, axis=0)
C_std_HRL_G_90 = np.std(C_avg_HRL_G_90, axis=0)
R_k_std = np.std(R_k_avg, axis=0)
C_k_std = np.std(C_k_avg, axis=0)
R_k_HRL_std = np.std(R_k_HRL_avg, axis=0)
C_k_HRL_std = np.std(C_k_HRL_avg, axis=0)

x = [i for i in range(len(R_mean_90))]
x1 = [i for i in range(len(R_k_mean))]

# legend = ["bvf-safe-sarsa", "hrl-safe-l-c-alloc", "hrl-safe-global-bvf", "unsafe_hrl"]
# legend = ["bvf-without_hrl", "global-bvf-with-hrl", "hrl-with-cost-allocation"]

legend = ["Unsafe-non-hrl","Unsafe-hrl", "BVF apporach", "HiLiTE", "Lyapunov apporach",]
# print(len(R_mean_60), len(R_mean_HRL_G_60))


fig, ax = plt.subplots(1, 1, figsize=(55, 35))
plt.tick_params(axis='both', which='major', labelsize=100)


#plt.plot(x, R_k_mean, label=legend[0], linewidth=10)
#plt.fill_between(x, R_k_mean + R_k_std, R_k_mean - R_k_std, alpha = 0.1)

plt.plot(x, R_k_HRL_mean, label=legend[1], linewidth=10)
plt.fill_between(x, R_k_HRL_mean + R_k_HRL_std, R_k_HRL_mean - R_k_HRL_std, alpha = 0.1)

plt.plot(x, R_mean_60, label=legend[2], linewidth=10)
plt.fill_between(x, R_mean_60 + R_std_60, R_mean_60 - R_std_60, alpha = 0.1)

plt.plot(x, R_mean_HRL_60, label=legend[3], linewidth=10)
plt.fill_between(x, R_mean_HRL_60 + R_std_HRL_60, R_mean_HRL_60 - R_std_HRL_60, alpha = 0.1)

plt.plot(x, R_lyp_mean_60, label=legend[4], linewidth=10)
plt.fill_between(x, R_lyp_mean_60 + R_lyp_std_60, R_lyp_mean_60 - R_lyp_std_60, alpha=0.1)

#plt.plot(x, R_mean_HRL_G_60, label=legend[2], linewidth=5)
#plt.fill_between(x, R_mean_HRL_G_60 + R_std_HRL_G_60, R_mean_HRL_G_60 - R_std_HRL_G_60, alpha = 0.1)

lgd = plt.legend( prop={'size':108},loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=False, ncol=3)
plt.xlabel('No of episodes X1000', size=90)
plt.ylabel("Reward", size=90)
plt.title("Reward comparision with a constriant limit of 30", size=90)
name = "figures/four_rooms/comp/Rewards_Key_30"
plt.savefig(name, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close(fig)


fig, ax = plt.subplots(1, 1, figsize=( 55, 35))
plt.tick_params(axis='both', which='major', labelsize=100)

#plt.plot(C_k_mean, label=legend[0], linewidth=8)
#plt.fill_between(x, C_k_mean + C_k_std, C_k_mean - C_k_std, alpha = 0.3)

plt.plot(C_k_HRL_mean, label=legend[1], linewidth=8)
plt.fill_between(x, C_k_HRL_mean + C_k_HRL_std, C_k_HRL_mean - C_k_HRL_std, alpha = 0.3)

plt.plot(C_mean_60, label=legend[2], linewidth=8)
plt.fill_between(x, C_mean_60 + C_std_60, C_mean_60 - C_std_60, alpha = 0.3)

plt.plot(C_mean_HRL_60, label=legend[3], linewidth=8)
plt.fill_between(x, C_mean_HRL_60 + C_std_HRL_60, C_mean_HRL_60 - C_std_HRL_60, alpha = 0.3)

plt.plot(C_lyp_mean_60, label=legend[4], linewidth=8)
plt.fill_between(x, C_lyp_mean_60 + C_lyp_std_60, C_lyp_mean_60 - C_lyp_std_60, alpha=0.3)

#plt.plot(C_mean_HRL_G_60, label=legend[2], linewidth=3)
#plt.fill_between(x, C_mean_HRL_G_60 + C_std_HRL_G_60, C_mean_HRL_G_60 - C_std_HRL_G_60, alpha = 0.3)

plt.axhline(y=30, color='r', linestyle='--',  linewidth=10)
lgd = plt.legend(prop={'size':108},loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=False, ncol=3)
plt.xlabel('No of episodes X1000', size=90)
plt.ylabel("Constraints", size=90)
plt.title("Constraint comparision with a constriant limit of 30", size=90)
name = "figures/four_rooms/comp/Constraints_Key_30"

ax.set_ylim(0, 350)
plt.savefig(name, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close(fig)




#stacked plot


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=( 110, 35))
ax1.tick_params(axis='both', which='major', labelsize=100)
ax2.tick_params(axis='both', which='major', labelsize=100)

ax1.plot(x, R_k_HRL_mean, label=legend[1], linewidth=10)
ax1.fill_between(x, R_k_HRL_mean + R_k_HRL_std, R_k_HRL_mean - R_k_HRL_std, alpha = 0.1)

ax1.plot(x, R_mean_60, label=legend[2], linewidth=10)
ax1.fill_between(x, R_mean_60 + R_std_60, R_mean_60 - R_std_60, alpha = 0.1)

ax1.plot(x, R_mean_HRL_60, label=legend[3], linewidth=10)
ax1.fill_between(x, R_mean_HRL_60 + R_std_HRL_60, R_mean_HRL_60 - R_std_HRL_60, alpha = 0.1)

ax1.plot(x, R_lyp_mean_60, label=legend[4], linewidth=10)
ax1.fill_between(x, R_lyp_mean_60 + R_lyp_std_60, R_lyp_mean_60 - R_lyp_std_60, alpha=0.1)

ax1.set_title("Reward comparision with a constriant limit of 30", size=90)
ax1.set_xlabel('No of episodes X1000', size=90)
ax1.set_ylabel("Reward", size=90)

ax2.plot(C_k_HRL_mean, label=legend[1], linewidth=8)
ax2.fill_between(x, C_k_HRL_mean + C_k_HRL_std, C_k_HRL_mean - C_k_HRL_std, alpha = 0.3)

ax2.plot(C_mean_60, label=legend[2], linewidth=8)
ax2.fill_between(x, C_mean_60 + C_std_60, C_mean_60 - C_std_60, alpha = 0.3)

ax2.plot(C_mean_HRL_60, label=legend[3], linewidth=8)
ax2.fill_between(x, C_mean_HRL_60 + C_std_HRL_60, C_mean_HRL_60 - C_std_HRL_60, alpha = 0.3)

ax2.plot(C_lyp_mean_60, label=legend[4], linewidth=8)
ax2.fill_between(x, C_lyp_mean_60 + C_lyp_std_60, C_lyp_mean_60 - C_lyp_std_60, alpha=0.3)


ax2.axhline(y=30, color='r', linestyle='--',  linewidth=10)
ax2.set_title("Constraint comparision with a constriant limit of 30", size=90)
ax2.set_xlabel('No of episodes X1000', size=90)
ax2.set_ylabel("Constraints", size=90)

ines_labels = [ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels()]
lines, labels = [sum(lol, []) for lol in zip(*ines_labels)]
unique = [(h, l) for i, (h, l) in enumerate(zip(lines, labels)) if l not in labels[:i]]
lines, labels = [], []
for i in unique:
    lines.append(i[0])
    labels.append(i[1])

lgd = fig.legend(lines, labels, prop={'size':108},loc='upper center', bbox_to_anchor=(0.5, -0.1),fancybox=True, shadow=False, ncol=5)
name = "figures/four_rooms/comp/Four_Rooms_30"
plt.savefig(name, bbox_extra_artists=(lgd,), bbox_inches='tight')
plt.close(fig)