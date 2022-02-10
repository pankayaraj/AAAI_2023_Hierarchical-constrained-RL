import torch
import numpy as np
import matplotlib.pyplot as plt

no_eps = 200000
l = no_eps//1000
a = []
for i in range(4):
    a.append(torch.load("results/grid/sarsa/c" + str(i+1))[:l])
for i in a:
    print(len(i))
""""""

no_experiments = 4
R = []
C = []

for i in range(no_experiments):
    R.append(torch.load("results/grid/safe_sarsa/r" + str(i+1))[:l])
    C.append(torch.load("results/grid/safe_sarsa/c" + str(i+1))[0:l])

R_avg = []
C_avg = []

for j in range(len(R)):
    r_avg = []
    c_avg = []
    for i in range(10,len(R[0])):
        r_avg.append(np.mean(R[j][i-10:i]))
        c_avg.append(np.mean(C[j][i-10:i]))

    R_avg.append(r_avg)
    C_avg.append(c_avg)

R_avg = np.array(R_avg)
C_avg = np.array(C_avg)

R_mean = np.mean(R_avg, axis=0)
C_mean = np.mean(C_avg, axis=0)

R_std = np.std(R_avg, axis=0)
C_std = np.std(C_avg, axis=0)


x = [i for i in range(len(R_mean))]
legend = ["bvf-safe-sarsa"]

fig, ax = plt.subplots(1, 1, figsize=(20, 10))
plt.tick_params(axis='both', which='major', labelsize=30)
plt.plot(x, R_mean, linewidth=5)
plt.fill_between(x, R_mean + R_std, R_mean - R_std, alpha = 0.3)
plt.legend(legend, prop={'size':40})
plt.xlabel('No of episodes X1000', size=40)
plt.ylabel("Reward", size=40)
plt.title("Reward comparision for the Grid World Environment", size=40)
name = "figures/grid/Rewards"
plt.savefig(name)
plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(20, 10))
plt.tick_params(axis='both', which='major', labelsize=30)
plt.plot(x, C_mean, linewidth=3)
plt.fill_between(x, C_mean + C_std, C_mean - C_std, alpha = 0.3)
plt.axhline(y=20, color='r', linestyle='--')
plt.legend(legend,prop={'size':40})
plt.xlabel('No of episodes X1000', size=40)
plt.ylabel("Constraints", size=40)
plt.title("Constraint comparision for the Grid World Environment", size=40)
name = "figures/grid/Constraints"

ax.set_ylim(0, 80)
plt.savefig(name)
plt.close(fig)
