# Created on 05/02/2025
# By Abheepsa Nanda
# To plot the percent of cases in which the 1.06 mutant emerges

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

num_seros = [1,2,3,5]
mut_rate = [1e-3]

colours = ['black', 'green', 'blue', 'red', 'purple', 'brown']

fig, ax = plt.subplots(figsize = (7,4))
for i in range(len(mut_rate)):
    for j in range(len(num_seros)):
        if num_seros[j]==1:
            filename = 'Summ_neg/fitnesstime_mut_rate_'+str(mut_rate[i])+'_numm_15_sero_'+str(num_seros[j])+'_g_5.0_sig_1e-12_m_-0.01.csv'
        else:
            filename = 'Summ_neg/fitnesstime_mut_rate_'+str(mut_rate[i])+'_numm_15_sero_'+str(num_seros[j])+'_g_5.0_sig_1e-12_m_-0.01.csv'
        data = pd.read_csv(filename, usecols = ['Times', 'Avged'], sep = ',')
        ax.plot(data['Times'][1:10], data['Avged'][1:10], color = colours[j], label = num_seros[j])

fig.legend(loc='center left', bbox_to_anchor=(1, 0.5), title = "Seros Num")
ax.grid()
plt.xlabel('Time')
plt.ylabel('Avg Population Fitness')
plt.title("Mutation rate = "+str(mut_rate[0])+', m = -0.01')
plt.tight_layout()
plt.savefig('Negative_fitness.png')
        
        