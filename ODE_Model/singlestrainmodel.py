# Created: 09/09/2024
# By: Abheepsa Nanda
# Code to model dynamics of spread of a beneficial mutation within a single strain

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

plt.rcParams.update({'font.size': 14})

#Initialising parameters
init_freq = np.array([0.9,0.1])
N = 1000
t = 1500
K = 10000
delta = 0.1
#free parameters
r = 1
m = 0.05*r

#differential equations
def onestrain(y, t, m, r):
    print(y)
    n1, n1m = y
    dydt = [n1 * r *(1-(n1+n1m)/K)-delta*n1, n1m * r * (1 + m)*(1-(n1+n1m)/K)-delta*n1m]
    return dydt

init_num = N*init_freq

t = np.linspace(0, t, 101)

sol = odeint(onestrain, init_num, t, args=(m, r))
freq = sol / sol.sum(axis=1)[:, None]

#plot the solution
labels = ["N1", "N1,m"]
fig, ax = plt.subplots(2,1,figsize=(6, 8))

ax[0].set_ylim(0,1.0)
ax[0].plot(t, freq[:,0], label="N1", linestyle = "dashed", color = 'tab:blue')
ax[0].plot(t, freq[:,1], label="N1,m", linestyle = "solid", color = 'tab:blue')
#ax.grid()
#ax.set_yscale('symlog')
ax[0].set_xlabel("Time", fontsize = 14)
ax[0].set_ylabel("Frequency", fontsize = 14)
ax[0].set_title('Number of serotypes = 1', fontsize = 16)
#ax.legend(loc = "center right", bbox_to_anchor =(1.,0.5) )
ax[0].legend(loc = 'upper right')

ax[1].set_ylim(0,1.0)
ax[1].plot(t, freq[:,1])
ax[1].set_xlabel("Time", fontsize = 14)
ax[1].set_ylabel("Mutant Frequency", fontsize = 14)
plt.tight_layout()
plt.savefig("Final_Images/SingleStrain.png")