# Created: 11/09/2024
# By: Abheepsa Nanda
# Code to model dynamics of spread of a beneficial mutation within multiple strains
# but here I'm also testing different parameters hence a different version was created.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#Assumptions: Equilibrium frequency is equal frequency of all strains.

#free parameters
r = 1
#r = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
vec_m = [0, 0.05 * r] #vec_m[0] = 0 since it corresponds to the individuals of the strain lacking a mutation
#vec_m = [0, 0.05]
#m_poss = [0.01 * r, 0.03 * r, 0.05 * r, 0.1 * r, 0.3 * r, 1 * r]
gamma = 0.1
#gamma = [0.01, 0.03, 0.1, 0.3, 0.5, 1, 3, 10]
sigma = 0.001
#sigma = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]

#Initialising parameters
#num_strains = 2
num_strains = [2,3,5,8,10, 15, 20]
N = 1000
#N = [100, 1000, 10000, 1e5]
time = 500
starting_mutant = 0.01
#starting_mutant = [0.001, 0.01, 0.1, 1]

#creating other variables
'''
freq_init = np.reshape(np.zeros((num_strains, 2)), (num_strains,2))
freq_init = np.zeros(num_strains * 2)
freq_init[0] = 1/num_strains - starting_mutant
freq_init[1:num_strains] = 1 / num_strains
freq_init[num_strains] = starting_mutant
freq_init[(num_strains+1):] = 0
numbers_init = freq_init * N

t = np.linspace(0, time, time * 1000)

#for graph labels
arr = list(range(num_strains))
arr = arr+arr
arr = [str(element) for element in arr]
arr[num_strains:] = [element + "m" for element in arr[num_strains:]]'''

#To make the code general, we have a function to generate the ODEs
def multistrain(y, t, vec_m, r, gamma, sigma, num_strains):
    numbers_mat = np.reshape(y,(num_strains,2), order= 'F')
    freq_mat = numbers_mat/np.sum(numbers_mat)
    diff_eqn = np.zeros((num_strains, 2))
    for i in range(0, num_strains):
        for j in [0, 1]:
            diff_eqn[i, j] = (numbers_mat[i, j] * r * (1 + gamma)**((1 / num_strains) - np.sum(freq_mat[i,:])) * (1 + vec_m[j])
                                - sigma * freq_mat[i, j] * (np.sum(freq_mat[:, (1 - j)]) - freq_mat[i, (1 - j)])
                                + sigma * freq_mat[i, (1 - j)] * (np.sum(freq_mat[:, j]) - freq_mat[i, j]))
    return diff_eqn.flatten('F')

def callmodel(init, timepoints, vector_m, r_growth, gamma_nfds, sigma_recom, number_strains):
    a = multistrain(init, timepoints, vector_m, r_growth, gamma_nfds, sigma_recom, number_strains)
    sol = odeint(multistrain, init, timepoints, args=(vector_m, r_growth, gamma_nfds, sigma_recom, number_strains))
    freq = sol / sol.sum(axis=1)[:, None]
    freq_mut = freq[:, number_strains:].sum(axis=1)
    time_maxmut = timepoints[np.argmax(freq_mut > 0.99)]
    return (sol, freq, time_maxmut)

#plot the solution
time_mut_fix = np.zeros(len(num_strains)) #flag

for i in range(len(num_strains)):  #flag
    freq_init = np.reshape(np.zeros((num_strains[i], 2)), (num_strains[i],2))
    freq_init = np.zeros(num_strains[i] * 2)
    freq_init[0] = 1/num_strains[i] - starting_mutant
    freq_init[1:num_strains[i]] = 1 / num_strains[i]
    freq_init[num_strains[i]] = starting_mutant
    freq_init[(num_strains[i]+1):] = 0
    numbers_init = freq_init * N

    t = np.linspace(0, time, time * 1000)

    #for graph labels
    arr = list(range(num_strains[i]))
    arr = arr+arr
    arr = [str(element) for element in arr]
    arr[num_strains[i]:] = [element + "m" for element in arr[num_strains[i]:]]
    sol, freq, time_mut_fix[i] = callmodel(numbers_init, t, vec_m, r, gamma, sigma, num_strains[i])#flag
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, freq, label = arr)
    #ax.plot(t, freq, label = range(10))
    #ax.set_yscale('symlog')
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency")
    ax.set_title("Multi-Strain Model - #strains = "+str(num_strains[i])) #flag
    ax.legend(loc = "lower right")
    plt.savefig('ODE_Model/num_strains/gamma0,1_num_strains_' + str(num_strains[i])+ ".png") #flag
    plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(num_strains, time_mut_fix, marker= 'o') #flag
#ax.plot(t, freq, label = range(10))
#ax.set_xscale('log')
ax.set_xlabel("Initial number of mutants")
ax.set_ylabel("Time till Fixation of Mutation")
ax.set_title("Two-Strain Model - Varying number of strains")
ax.grid()
#ax.legend(loc = "lower right")
plt.savefig("ODE_Model/num_strains/gamma0,1_fixation_based_on_number_of_strains.png")