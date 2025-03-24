# Created on: 13/12
# By: Abheepsa Nanda
# Code to implement stochastic model with tau leaping - for running on Euler
# Code is for a single mutation
# This version implements stopping once double mutant has fixed and recording a lower number of points
# Also, this version implements a faster algorithm wherein only small
# number calculations are done with randomization
# Changes from v1 to make code faster:
# 1. skipping forward during stable regions using a Gillespie algorithm to 
#    calculate time of next recombination event - done
# 2. increasing dt
# 3. preventing useless recombinations - have not done yet

import numpy as np
from scipy.stats import poisson
from scipy.stats import expon
import argparse
import os
import pandas as pd

#to help accept a list of values as arguments for init_muts
def list_of_ints(arg):
    return list(map(int, arg.split(',')))

#to help accept a list of values as arguments for mutation benefits
def list_of_floats(arg):
    return list(map(float, arg.split(',')))

#function to generate column names for the time series dataframe
def columnnames(num_muts, num_seros):
    headers = ['Time']
    for i in range(num_seros):
        for j in range(2**num_muts):
            geno = bin(j)[2:]
            str_geno = '0'*(num_muts-len(geno))+geno
            str_name = 'N'+str(i)+'_'+str_geno
            headers += [str_name]
    return headers
    

#Creating a multidimensional array of presence/absence matrices of mutations
def create_mut_tensor(num_mut, num_sero):
    mut_tensor = np.zeros((num_mut, num_sero, 2**num_mut))
    for i in range(num_mut):
        temp = np.zeros(2 ** num_mut)
        for j in range(2 ** num_mut):
            if ((int)(j / 2 ** i) % 2) == 1:
                temp[j] = 1
        temp_mat = np.reshape(np.repeat(temp, num_sero), (num_sero, 2 ** num_mut), order ='F')
        mut_tensor[i] = temp_mat
    return mut_tensor

#Function to take two recombining species, locus of recombination and give the resulting recombinants
def get_recombinants(k,ind1, ind2, mut_tensor, num_mutations):
    if (mut_tensor[k, ind1[0], ind1[1]]==mut_tensor[k, ind2[0], ind2[1]]):
        return -1 #this is the case where the two genotypes do not differ at the locus k
    mutn_slice_i = np.copy(mut_tensor[:,ind1[0],ind1[1]])
    mutn_slice_i[k] = 1-mut_tensor[k,ind1[0],ind1[1]]
    new_ind_1 = [ind1[0],int(np.sum(np.multiply(mutn_slice_i, 2**np.arange(0,num_mutations))))]
    mutn_slice_a = np.copy(mut_tensor[:,ind2[0],ind2[1]])
    mutn_slice_a[k] = 1-mut_tensor[k,ind2[0],ind2[1]]
    new_ind_2 = [ind2[0],int(np.sum(np.multiply(mutn_slice_a, 2**np.arange(0,num_mutations))))]
    return new_ind_1, new_ind_2

#calculate differences for numerical simulation - switches between deterministic and random regimes
def difference_terms(numbers_mat, num_serotypes, num_mutations, K, r, gamma, m, sigma, delta, mut_tensor, thres, tau):
    freqs = numbers_mat / np.sum(numbers_mat)
    strainfreq = np.sum(freqs, axis = 1)
    strainfreq_mat = np.reshape(np.repeat(strainfreq, 2**num_mutations), (num_serotypes, 2**num_mutations))
    growth_diff = tau*r *(1-np.sum(numbers_mat)/K)*np.multiply(np.multiply(numbers_mat, 1+np.einsum('i,ijk->jk', m, mut_tensor)), (1 + gamma)**(- strainfreq_mat))
    death_diff = -1*tau*delta*numbers_mat
    recomb_diff = np.zeros((num_serotypes, 2 ** num_mutations))
    for i in range(num_serotypes):
        for j in range(2 ** num_mutations):
            if numbers_mat[i,j]<thres:
                growth_diff[i,j] = np.multiply(np.sign(growth_diff[i,j]), poisson.rvs(mu = np.absolute(growth_diff[i,j])))
                death_diff[i,j] = -1*poisson.rvs(mu = np.absolute(death_diff[i,j]))
            for a in range(i, num_serotypes):
                for b in range(2**num_mutations):
                    if j==b:
                        continue
                    for k in range(num_mutations):
                        mutn = np.array(mut_tensor[k])
                        if mutn[i,j] == mutn[a,b]:
                            continue
                        change = poisson.rvs(sigma[k]*numbers_mat[i,j]*numbers_mat[a,b]*tau)
                        if change>0:
                            recomb_diff[i,j] += -change
                            recomb_diff[a,b] += -change
                            newinds = get_recombinants(k, (i,j), (a,b), mut_tensor, num_mutations)
                            recomb_diff[newinds[0][0], newinds[0][1]] += change
                            recomb_diff[newinds[1][0], newinds[1][1]] += change
    growth_diff = np.rint(growth_diff)
    death_diff = np.rint(death_diff)
    return(growth_diff, recomb_diff, death_diff)

#calculates waiting time till next recombination event when system hits a stable point
def gillespie_time_jump(numbers_mat, num_serotypes, num_mutations, sigma,mut_tensor):
    comb1 = list()
    comb2 = list()
    mutation_ind = list()
    propensity = list()
    for i in range(num_serotypes):
        for j in range(2**num_mutations):
            for a in range(i, num_serotypes):
                for b in range(2**num_mutations):
                    for k in range(num_mutations):
                        if numbers_mat[i,j] == 0 or numbers_mat[a,b] == 0:
                            continue
                        if mut_tensor[k,i,j]!=mut_tensor[k,a,b]:
                            comb1 += [i*2**num_mutations+j]
                            comb2 += [a*2**num_mutations+b]
                            mutation_ind += [k]
                            propensity += [sigma[k]*numbers_mat[i,j]*numbers_mat[a,b]]
    if (np.sum(propensity) == 0.0):
        #this means that no recombination is possible - this model cannot proceed further
        #for example, if only single mutants of a particular mutation exist and the other mutation
        #has been wiped out
        return -1,-1, -1
    wait_time = float(round(expon.rvs(scale = 1/np.sum(np.array(propensity, dtype = float))),4))
    chosen_rxn = np.random.choice(np.arange(len(propensity), dtype = int), 1, p = np.array(propensity)/np.sum(np.array(propensity)))
    chosen_rxn = chosen_rxn[0]
    updated_numbers_mat = np.copy(numbers_mat)
    chosen_comb1 = [(int)(comb1[chosen_rxn]/2**num_mutations), (int)(comb1[chosen_rxn]%(2**num_mutations))]
    chosen_comb2 = [(int)(comb2[chosen_rxn]/2**num_mutations), (int)(comb2[chosen_rxn]%(2**num_mutations))]
    updated_numbers_mat[chosen_comb1[0], chosen_comb1[1]] -=1
    updated_numbers_mat[chosen_comb2[0], chosen_comb2[1]] -=1
    recomb_inds = get_recombinants(mutation_ind[chosen_rxn], chosen_comb1, chosen_comb2, mut_tensor, num_mutations)
    updated_numbers_mat[recomb_inds[0][0], recomb_inds[0][1]] +=1
    updated_numbers_mat[recomb_inds[1][0], recomb_inds[1][1]] +=1
    return updated_numbers_mat, wait_time, 0

#implements tau leaping algorithm
def tau_leaping(num_serotypes, num_mutations, init_muts, K, time_end, tau, r, gamma, m, sigma, delta, numbers_init, thres):
    tosave = pd.DataFrame(columns = columnnames(num_mutations, num_serotypes))
    tosave.loc[0] = [0]+numbers_init.flatten().tolist()
    numbers_series = np.zeros(((int)(time_end/tau)+1, num_serotypes, 2**num_mutations))
    numbers_series[0] = numbers_init
    mut_tensor = create_mut_tensor(num_mutations, num_serotypes)
    time = 0
    i=1
    counter_overextinction = 0
    gillespie = False
    while time<time_end:
        if gillespie == True:
            changed, tjump, flag = gillespie_time_jump(numbers_series[i-1], num_serotypes, num_mutations, sigma, mut_tensor)
            if flag == -1: 
                #this means that no recombination is possible - this model cannot proceed further
                #for example, if only single mutants of a particular mutation exist and the other mutation
                #has been wiped out
                return tosave, counter_overextinction
            numbers_series[i] = changed
            time = time + tjump
            gillespie = False
            neg_indices = np.where(numbers_series[i]<0)
            numbers_series[i][neg_indices] = 0
            tosave.loc[tosave.shape[0]] = [time]+ numbers_series[i].flatten().tolist()
            i= i+1
        elif gillespie == False:
            growth_diff, recomb_diff, death_diff = difference_terms(numbers_series[i-1], num_serotypes, num_mutations, K, r, gamma, m, sigma, delta, mut_tensor, thres, tau)
            numbers_series[i] = numbers_series[i-1]+growth_diff+recomb_diff+death_diff
            counter_overextinction += np.count_nonzero(numbers_series[i]<0)
            neg_indices = np.where(numbers_series[i]<0)
            numbers_series[i][neg_indices] = 0
            time = time+tau
            if i%5==0:
                tosave.loc[tosave.shape[0]] = [time]+ numbers_series[i].flatten().tolist()
            
            # stopping condition - if double mutants have taken over the population
            if (np.sum(numbers_series[i, :, 2**(num_mutations)-1])>np.sum(numbers_series[i])*0.9999):
                return tosave, counter_overextinction
            
            #when system hits a stable point where numbers remain constant, only a recombination can move it from this
            #stable point and waiting times for such recombinations are long so we do a time jump using gillespie algorithm
            if (i>10 and np.array_equal(numbers_series[i],numbers_series[i-10])==True):
                gillespie = True
            i = i+1
    return tosave, counter_overextinction

def main():
    parser = argparse.ArgumentParser(description="Accepts arguments from command line")

    #accept arguments
    parser.add_argument("--num_serotypes", type=int,  help="Initial number of serotypes that exist in the population", default = 1)
    parser.add_argument("--num_mutations", type=int, help = "Number of mutations in the population", default= 2)
    parser.add_argument("--init_muts", type=list_of_ints, help = "Serotypes in which mutants are initially found (0, 1, ..., num_seros-1)", default = [0,0])
    parser.add_argument("--K", type=int, help = "Carrying capacity", default = 100000)
    parser.add_argument("--time", type=int, help = "Time period of simulation", default = 2000000)
    parser.add_argument("--tau", type=float, help = "Time step size for tau leaping algorithm", default = 2)
    parser.add_argument("--r", type=float, help="Base growth rate", default = 1)
    parser.add_argument("--gamma", type = float, help ="Strength of NFDS", default = 0.5)
    parser.add_argument("--m", type=list_of_floats, help = "Benefit due to mutation", default = -1)
    parser.add_argument("--sigma", type = list_of_floats, help = "Recombination rate", default=-1) #recombination constant
    parser.add_argument("--delta", type = float, help = "Death rate", default = 0.1)
    parser.add_argument("--iter", type = int, help = "Repeat number", default = 0)
    parser.add_argument("--random_thres", type = int, help = "Threshold number of individuals below which randomization is applied", default = 100)
   
    #assign arguments to variables
    args = parser.parse_args()
    num_serotypes = args.num_serotypes
    num_mutations = args.num_mutations
    init_muts = args.init_muts
    if init_muts==-1:
        init_muts = [0]*num_mutations
    if len(init_muts)!=num_mutations:
        print("Terminated: Number of starting compartments for mutations is not the same as the number of mutations")
        return
    K = args.K
    time_end = args.time
    tau = args.tau
    r = args.r
    gamma = args.gamma
    m = args.m
    if m==-1:
        m = [0.05]*num_mutations
    if len(m)!=num_mutations:
        print("Terminated: Number of values of m is not the same as the number of mutations")
        return
    sigma = args.sigma
    if sigma==-1:
        sigma = [1e-12]*num_mutations
    if len(sigma)!=num_mutations:
        print("Terminated: Number of values of sigma is not the same as the number of mutations")
        return
    delta = args.delta
    iter = args.iter
    thres = args.random_thres
    numbers_init = np.zeros((num_serotypes, 2 ** num_mutations))
    numbers_init[:, 0] = (int)(K/(2*num_serotypes)) #sets WT strains numbers to 100
    for i in range(len(init_muts)): 
        numbers_init[init_muts[i], 2 ** i] = (int)(K*0.01/(2*num_serotypes)) #sets number of mutants for each locus to 1
    #check if directory for output files exists and if it doesn't, creates it
    if (os.path.exists('Output')==False):
        os.mkdir('Output')
    #run model
    tosave, counter_overext = tau_leaping(num_serotypes, num_mutations, init_muts, K, time_end, tau, r, gamma, m, sigma, delta, numbers_init, thres)
    #save run to csv
    tosave.to_csv('Output/numm_'+str(num_mutations)+'_sero_'+str(num_serotypes)+'_initm_'+str(init_muts)+'_g_'+str(gamma)+'_sig_'+str(sigma)+'_m_'+str(m)+'_iter_'+str(iter)+'.csv', sep = ',', index=False)

if __name__ == "__main__":
    main()