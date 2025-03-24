# Created on 21/10/2024
# By Abheepsa Nanda
# Code to take run files and create csvs of files with fixation times and serotypes left
# at fixation time for all iterations of a particular set of conditions

import pandas as pd
import numpy as np
import os

num_muts = 2
num_seros = [2,3,4]
init_muts = [[0,1]]
gamma = [10.0]
num_iter = 100
recomb_rate = [[1e-13,1e-13]]
mut = [[0.07, 0.07]]

# function returns the time until a mutant fixes and which genotype fixes
def time_fixation(data, time):
    last_state = data[-1]
    last_state_freqs = np.sum(last_state/np.sum(last_state), axis = 0)
    for i in range(4):
        if (last_state_freqs[i]>0.99):
            return i, time[-1]
    return -1

#function returns fixation time of mutation - could be single mutation or double mutant - I suggest adding
#an additional return value for which fixed at the end
def seros_end(data):
    last_times=np.sum(data[-1], axis = 1)
    counter = 0
    for i in range(last_times.shape[0]):
        if last_times[i]>1:
            counter = counter+1
    return counter

#function returns time of emergence of double mutant - it's the first timepoint where
#number of individuals of double mutant become >500 in any strain because then it will
#definitely grow in number - return -1 if double mutant never emerged
def emerge_doublemut(data, time):
    doublemuts = data[:,:,3]
    if (np.where(doublemuts>500)[0]).shape[0]==0:
        return -1
    timing = np.where(doublemuts >500)[0][0]
    return time[timing]

        
for i in num_seros:
    for j in init_muts:
        for k in gamma:
            for l in recomb_rate:
                for m in mut:
                    time_fix_muts = pd.DataFrame(columns = ['Time_fix', 'Genotype'])
                    num_seros_end = pd.DataFrame(columns = ['Seros_left'])
                    emergence_doub = pd.DataFrame(columns = ['T_emerge'])
                    instance = 'numm_2_sero_'+str(i)+'_initm_'+str(j)+'_g_'+str(k)+'_sig_'+str(l)+'_m_'+str(m)
                    for p in range(num_iter):
                        filename = 'Math_Comp/numm_2_sero_'+str(i)+'_initm_'+str(j)+'_g_'+str(k)+'_sig_'+str(l)+'_m_'+str(m)+'_iter_'+str(p)+'.csv'
                        if (os.path.exists(filename) == True):
                            data = pd.read_csv(filename, delimiter = ',')
                            time = np.array(data.loc[:,'Time'])
                            numms = np.reshape(np.array(data.loc[:,data.columns!='Time']), (data.shape[0], i, 4))
                            time_fix_muts.loc[p] = time_fixation(numms, time)
                            num_seros_end.loc[p] = seros_end(numms)
                            emergence_doub.loc[p] = emerge_doublemut(numms, time)
                        else:
                            print(filename)
                    time_fix_muts.to_csv('Summary/Time_fixation/tfix_'+instance+'.csv',sep = ',', index=False)
                    num_seros_end.to_csv('Summary/Seros_end/serosend_'+instance+'.csv',sep = ',', index=False)
                    emergence_doub.to_csv('Summary/Emerge_doublemut/emer_double_'+instance+'.csv',sep = ',', index=False)