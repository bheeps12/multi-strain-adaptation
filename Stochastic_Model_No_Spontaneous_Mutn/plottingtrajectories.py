# Created on 08-10-2024
# By Abheepsa Nanda
# Code to take a csv file of a run and plot the trajectories

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

num_muts = 2
num_seros = 3
colours = ('red', 'green', 'blue', 'purple', 'orange', 'pink', 'orange', 'brown', 'black', 'grey')
styles_of_line = ('dotted', 'dashed', 'dashdot', 'solid')
filename = 'Plotted Dynamics/numm_2_sero_3_initm_[0, 1]_g_1.0_sig_[1e-12, 1e-12]_m_[0.02, 0.05]_iter_96.csv'
data = pd.read_csv(filename, delimiter = ',')
fig, ax = plt.subplots(figsize=(8, 5))
for i in range(num_seros):
    for j in range(2**num_muts):
        ax.plot(data.iloc[:,0], data.iloc[:,i*2**num_muts+j+1], label =  data.iloc[:,i*2**num_muts+j+1].name, color = colours[i], linestyle = styles_of_line[j])
ax.set_xlabel("Time")
ax.set_ylabel("Numbers")
ax.set_title(str(num_seros)+" serotypes, 2 mutations with Large Fitness Difference")
ax.legend(loc = 'center left',bbox_to_anchor=(1, 0.5))
ax.grid()
plt.tight_layout()
plt.savefig('Traj_diffsero_largediff.png')
