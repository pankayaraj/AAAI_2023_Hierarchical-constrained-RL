import torch
import numpy as np
import matplotlib.pyplot as plt

no_eps = 40000



no_experiments = 4
R = []
C = []

for i in range(no_experiments):
    R.append(torch.load("results/grid/safe_sarsa/r" + str(i+1)))
    C.append(torch.load("results/grid/safe_sarsa/c" + str(i+1)))

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

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.plot(x, R_mean, linewidth=3)
plt.fill_between(x, R_mean + R_std, R_mean - R_std, alpha = 0.3)
#plt.legend(legend)
plt.xlabel('No of episodes X1000')
plt.ylabel("Reward")
plt.title("Rewards for the Grid World Environment")
name = "figures/grid/Rewards"
plt.savefig(name)
plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.plot(x, C_mean, linewidth=3)
plt.fill_between(x, C_mean + C_std, C_mean - C_std, alpha = 0.3)
plt.axhline(y=20, color='r', linestyle='--')
#plt.legend(legend)
plt.xlabel('No of episodes X1000')
plt.ylabel("Constraints")
plt.title("Constraints for the Grid World Environment")
name = "figures/grid/Constraints"
plt.savefig(name)
plt.close(fig)