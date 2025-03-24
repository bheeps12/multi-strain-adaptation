import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

num_muts = 2
num_seros = [1, 2,3,5,8,10]
init_muts = [[0,1]]
gamma = [0.1, 0.5, 1.0]
sigma = [[1e-13,1e-13],[1e-12,1e-12], [1e-11,1e-11]]
mut = [[0.02,0.05], [0.04,0.05], [0.05,0.05]]

fig, ax = plt.subplots(3,3, figsize = (25,20))
fig.suptitle('Emergence of Double Mutant', y = 0.93, fontsize = 20)
colours_lines = ['green', 'blue', 'red', 'purple']
r = 0
c = 0
lines = []
for j in init_muts:
    r=0
    for k in gamma:
        c=0
        for l in sigma:
            another_counter = 0
            lines= []
            for m in mut:
                print(r)
                print(c)
                #ax[r,c].set_ylim((0,1))
                ax[r,c].set_xlabel("Initial Number of Serotypes")
                ax[r,c].set_ylabel("Percent of cases with double mutant fixn") 
                ax[r,c].set_title(r'$\gamma$ = '+str(k)+r', $\sigma$ ='+str(l), fontsize = 14)
                ax[r,c].grid()
                counter = 0
                fractions = np.zeros(len(num_seros))
                for i in num_seros:
                    if (i == 1):
                        instance = 'emer_double_'+'numm_2_sero_'+str(i)+'_initm_[0, 0]_g_0.0_sig_'+str(l)+'_m_'+str(m)
                    else:
                        instance = 'emer_double_'+'numm_2_sero_'+str(i)+'_initm_'+str(j)+'_g_'+str(k)+'_sig_'+str(l)+'_m_'+str(m)
                    dat = np.array(pd.read_csv('Summary/Emerge_doublemut/'+instance+'.csv', usecols = ['T_emerge']))
                    #dat = dat-3
                    #doub_count = dat.shape[0]-np.count_nonzero(dat)
                    #fractions[counter]=doub_count/dat.shape[0]
                    #fractions[counter] = np.median(dat)
                    counter =counter+1
                lines+=ax[r,c].plot(num_seros, fractions, color = colours_lines[another_counter])
                another_counter +=1
            c=c+1
        r=r+1
legend = fig.legend(lines, mut, title = "m", loc = "center right", fontsize = 16)
plt.setp(legend.get_title(),fontsize=18)
                    
                    
                    
                    
                    