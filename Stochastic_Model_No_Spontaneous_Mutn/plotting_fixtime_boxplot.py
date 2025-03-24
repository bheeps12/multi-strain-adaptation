# Created on 09-10-2024
# By Abheepsa
# Code to take in csv's of time until fixation of mutant of different runs
# saved as 'Time_fix_gamma_x.csv' and plot them.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

num_muts = 2
num_seros = [2,3,5,8,10]
init_muts = [[0,0]]
gamma = [0.1, 0.5, 1.0]
sigma = [[1e-13,1e-13],[1e-12,1e-12], [1e-11,1e-11]]
mut = [[0.05,0.05]]

fig, ax = plt.subplots(3,3, figsize = (25,20))
fig.suptitle('Time until fixation of mutation with m='+str(mut[0])+' starting in same compartments', y = 0.93, fontsize = 25)
fig.supxlabel("Initial Number of Serotypes", fontsize = 25, y = 0.06)
fig.supylabel("Fixation time", fontsize = 25, x = 0.08)
colours_lines = ['green', 'blue', 'red', 'purple']
r = 0
c = 0
lines = []
for m in mut:
    for j in (init_muts):
        for k in range(len(gamma)):
            for l in range(len(sigma)):
                another_counter = 0
                lines= []
                #ax[j,k].set_ylim((0,1))
                #ax[k,l].set_xlabel("Initial Number of Serotypes", fontsize = 20)
                #ax[k,l].set_ylabel("Fixation time", fontsize = 20) 
                ax[k,l].set_xticklabels(num_seros, fontsize =25)
                ax[k,l].set_title(r'$\gamma$ = '+str(gamma[k])+r', $\sigma$ ='+str(sigma[l]), fontsize = 20)
                ax[k,l].grid()
                counter = 0
                tfix = pd.DataFrame(columns = num_seros, index=np.arange(100), dtype=float)
                for i in num_seros:
                    instance = 'tfix_'+'numm_2_sero_'+str(i)+'_initm_'+str(j)+'_g_'+str(gamma[k])+'_sig_'+str(sigma[l])+'_m_'+str(m)
                    dat = (pd.read_csv('Summary/Time_fixation/'+instance+'.csv')).to_numpy(dtype = float)
                    dat = dat[:,1]
                    if dat.shape[0]<100:
                        dat = np.hstack((dat,[np.nan]*(100-dat.shape[0])))
                    tfix.loc[:,i] = dat
                    counter =counter+1
                #ax[k,l].set_yticklabels(labels = ax[k,l].get_yticklabels(), fontsize =15)
                #ax[k,l].boxplot(tfix, flierprops=dict(markeredgecolor='none', markerfacecolor='none', markersize=0))
                ax[k,l].boxplot(tfix, tick_labels = num_seros, showfliers = False)
                #ax[k,l].violinplot(tfix)
                another_counter +=1
    #legend = fig.legend(lines, mut, title = "m", loc = "center right", fontsize = 16)
    #plt.setp(legend.get_title(),fontsize=18)
