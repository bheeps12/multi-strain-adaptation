# Created on: 17/01
# By: Abheepsa Nanda
# Code to implement stochastic model with tau leaping - for running on Euler
# Code has spontaneous mutations arising with m = 0.01
# This version implements stopping once double mutant has fixed and recording a lower number of points
# Also, this version implements a faster algorithm wherein only small
# number calculations are done with randomization
# This version writes results in chunks to free up memory

import numpy as np
from scipy.stats import poisson
from scipy.stats import expon
import argparse
import os
import pandas as pd
#import time
import sys

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

def get_mutant(k, ind1, mut_tensor, num_mutations):
    mutn_slice_i = np.copy(mut_tensor[:,ind1[0],ind1[1]])
    mutn_slice_i[k] = 1
    new_ind_1 = [ind1[0],int(np.sum(np.multiply(mutn_slice_i, 2**np.arange(0,num_mutations))))]
    return new_ind_1

def num_of_mutations(ind):
    return np.count_nonzero(np.array(list(bin(ind)[2:]), dtype = int))

#calculate differences for numerical simulation - switches between deterministic and random regimes
def difference_terms(numbers_mat, num_serotypes, num_mutations, K, r, mut_rate, gamma, m, sigma, delta, mut_tensor, thres, tau):
    freqs = numbers_mat/np.sum(numbers_mat)
    strainfreq = np.sum(freqs, axis = 1)
    strainfreq_mat = np.reshape(np.repeat(strainfreq, 2**num_mutations), (num_serotypes, 2**num_mutations))
    growth_diff = tau*r *(1-np.sum(numbers_mat)/K)*np.multiply(np.multiply(numbers_mat, 1+m*np.sum(mut_tensor, axis = 0)), (1 + gamma)**((1/num_serotypes)- strainfreq_mat))
    death_diff = -1*tau*delta*numbers_mat
    recomb_diff = np.zeros((num_serotypes, 2 ** num_mutations))
    mut_diff = np.zeros((num_serotypes, 2 ** num_mutations))
    mut_poss = poisson.rvs(mu = numbers_mat*mut_rate)
    counter = 0
    non_zero_indices = np.array(np.where(numbers_mat>0)).T
    if np.sum(mut_poss)>0:
        mut_inds = np.where(mut_poss>0)
        for i in range(len(mut_inds[0])):
            ind = np.random.choice(num_mutations)
            if mut_tensor[ind, mut_inds[0][i], mut_inds[1][i]] ==1:
                counter +=1
                continue
            mut_diff[mut_inds[0][i], mut_inds[1][i]] = -1
            mutant = get_mutant(ind, [mut_inds[0][i], mut_inds[1][i]], mut_tensor, num_mutations)
            mut_diff[mut_inds[0][i], mutant[1]] = 1
    for i in non_zero_indices:
        if numbers_mat[i[0],i[1]]<thres and numbers_mat[i[0],i[1]]>0:
            growth_diff[i[0],i[1]] = np.multiply(np.sign(growth_diff[i[0],i[1]]), poisson.rvs(mu = np.absolute(growth_diff[i[0],i[1]])))
            death_diff[i[0],i[1]] = -1*poisson.rvs(mu = np.absolute(death_diff[i[0],i[1]]))                    
        for j in non_zero_indices:
                if (i==j).all():
                    continue
                for k in range(num_mutations):
                    if mut_tensor[k,i[0], i[1]] == mut_tensor[k, j[0], j[1]]:
                        continue
                    change = poisson.rvs(sigma*numbers_mat[i[0],i[1]]*numbers_mat[j[0],j[1]]*tau)
                    if change>0:
                        recomb_diff[i[0],i[1]] += -change
                        recomb_diff[j[0],j[1]] += -change
                        newinds = get_recombinants(k, i, j, mut_tensor, num_mutations)
                        recomb_diff[newinds[0][0], newinds[0][1]] += change
                        recomb_diff[newinds[1][0], newinds[1][1]] += change
    return(growth_diff, recomb_diff, death_diff, mut_diff, counter)

#calculates waiting time till next recombination event when system hits a stable point
def gillespie_time_jump(numbers_mat, num_serotypes, num_mutations, sigma,mut_tensor, mut_rate):
    comb1 = list()
    comb2 = list()
    mutation_ind = list()
    propensity = list()
    prob_of_fixation = 0.02
    non_zero_indices = np.array(np.where(numbers_mat>0)).T
    for i in non_zero_indices:
        chosen_spot = np.random.choice(num_mutations)
        mut_result = get_mutant(chosen_spot, i, mut_tensor, num_mutations)
        if numbers_mat[mut_result[0], mut_result[1]]==0:
            propensity += [numbers_mat[i[0],i[1]]*mut_rate*prob_of_fixation]
            comb1 += [i[0]*2**num_mutations+i[1]]
            comb2 += ['na']
            mutation_ind+=[chosen_spot]
        for j in non_zero_indices:
            for k in range(num_mutations):
                if mut_tensor[k,i[0],i[1]]!=mut_tensor[k,j[0],j[1]]:
                    comb1 += [i[0]*2**num_mutations+i[1]]
                    comb2 += [j[0]*2**num_mutations+j[1]]
                    mutation_ind += [k]
                    propensity += [sigma*numbers_mat[i[0],i[1]]*numbers_mat[j[0],j[1]]]
    if (np.sum(propensity) == 0.0):
        #this means that no recombination is possible - this model cannot proceed further
        #for example, if only single mutants of a particular mutation exist and the other mutation
        #has been wiped out
        return -1,-1, -1, 0
    wait_time = float(round(expon.rvs(scale = 1/np.sum(np.array(propensity, dtype = float))),4))
    chosen_rxn = np.random.choice(np.arange(len(propensity), dtype = int), 1, p = np.array(propensity)/np.sum(np.array(propensity)))
    chosen_rxn = chosen_rxn[0]
    updated_numbers_mat = np.copy(numbers_mat)
    chosen_comb1 = [(int)(comb1[chosen_rxn]/2**num_mutations), (int)(comb1[chosen_rxn]%(2**num_mutations))]
    if comb2[chosen_rxn] == 'na':
        if mut_tensor[mutation_ind[chosen_rxn], chosen_comb1[0], chosen_comb1[1]] == 1:
            return updated_numbers_mat, wait_time, 0, 1
        else:
            updated_numbers_mat[chosen_comb1[0], chosen_comb1[1]] -=100
            mutated_ind = get_mutant(mutation_ind[chosen_rxn], chosen_comb1, mut_tensor, num_mutations)
            updated_numbers_mat[mutated_ind[0], mutated_ind[1]] +=100
            return updated_numbers_mat, wait_time, 0, 0
    else:
        chosen_comb2 = [(int)(comb2[chosen_rxn]/2**num_mutations), (int)(comb2[chosen_rxn]%(2**num_mutations))]
        updated_numbers_mat[chosen_comb1[0], chosen_comb1[1]] -=1
        updated_numbers_mat[chosen_comb2[0], chosen_comb2[1]] -=1
        recomb_inds = get_recombinants(mutation_ind[chosen_rxn], chosen_comb1, chosen_comb2, mut_tensor, num_mutations)
        updated_numbers_mat[recomb_inds[0][0], recomb_inds[0][1]] +=1
        updated_numbers_mat[recomb_inds[1][0], recomb_inds[1][1]] +=1
        return updated_numbers_mat, wait_time, 0, 0
    print("Something went wrong")

#implements tau leaping algorithm
def tau_leaping(num_serotypes, num_mutations, K, time_end, tau, r, mut_rate, gamma, m, sigma, delta, numbers_init, thres, output_filename):
    vec_num_mutations = np.vectorize(num_of_mutations)
    tosave = pd.DataFrame(columns = columnnames(num_mutations, num_serotypes))
    tosave.loc[0] = [0]+numbers_init.flatten().tolist()
    tosave.to_csv(output_filename, sep = ',', index = False)
    numbers_series = np.zeros((11, num_serotypes, 2**num_mutations))
    numbers_series[9] = numbers_init
    mut_tensor = create_mut_tensor(num_mutations, num_serotypes)
    time = 0
    j = 0
    counter_overextinction = 0
    counter_mutsame = 0
    gillespie = False
    while time<time_end:
        if gillespie == True:
            changed, tjump, flag, counterm = gillespie_time_jump(numbers_series[9], num_serotypes, num_mutations, sigma, mut_tensor, mut_rate)
            if flag == -1: 
                #this means that no recombination is possible - this model cannot proceed further
                #for example, if only single mutants of a particular mutation exist and the other mutation
                #has been wiped out
                # in current code this should never be achieved
                print("This condition 3")
                return tosave, counter_overextinction, counter_mutsame
            counter_mutsame += counterm
            numbers_series[10] = changed
            time = time + tjump
            gillespie = False
            neg_indices = np.where(numbers_series[10]<0)
            numbers_series[10][neg_indices] = 0
            tosave.loc[0] = [time]+ numbers_series[10].flatten().tolist()
            tosave.to_csv(output_filename, header = None, mode = "a", sep = ',', index = False)
            numbers_series[0:10] = numbers_series[1:]
        elif gillespie == False:
            growth_diff, recomb_diff, death_diff, mut_diff, counterm = difference_terms(numbers_series[9], num_serotypes, num_mutations, K, r, mut_rate, gamma, m, sigma, delta, mut_tensor, thres, tau)
            numbers_series[10] = numbers_series[9]+growth_diff+recomb_diff+death_diff+mut_diff
            counter_mutsame += counterm
            counter_overextinction += np.count_nonzero(numbers_series[10]<0)
            neg_indices = np.where(numbers_series[10]<0)
            numbers_series[10][neg_indices] = 0
            time = time+tau
            if j%100==0:
                tosave.loc[0] = [time]+ numbers_series[10].flatten().tolist()
                tosave.to_csv(output_filename, header = None, mode = "a", sep = ',', index = False)
                j = 0
            j = j+1
            numbers_series[0:10] = numbers_series[1:]
            # stopping condition - if double mutants have taken over the population - this won't happen
            # anymore
            #if (np.sum(numbers_series[9, :, 2**(num_mutations)-1])>np.sum(numbers_series[10])*0.9999):
            #    return tosave, counter_overextinction, counter_mutsame
            
            # stopping condition - if more than half of the sites are mutated
            geno_present = np.array(np.where(np.sum(numbers_series[9], axis = 0)>0)[0])
            curr_mutants = np.array(vec_num_mutations(geno_present))
            if np.any(curr_mutants> (int)(num_mutations/2)):
                print("This condition 1")
                return tosave, counter_overextinction, counter_mutsame
            
            #when system hits a stable point where numbers remain constant, only a recombination can move it from this
            #stable point and waiting times for such recombinations are long so we do a time jump using gillespie algorithm
            if (np.array_equal(numbers_series[9],numbers_series[0])==True):
                gillespie = True
    print("This condition 2")
    return tosave, counter_overextinction, counter_mutsame

def main():
    #start= time.time()
    parser = argparse.ArgumentParser(description="Accepts arguments from command line")

    #accept arguments
    parser.add_argument("--num_serotypes", type=int,  help="Initial number of serotypes that exist in the population", default = 2)
    parser.add_argument("--init_num_mutations", type=int, help = "Number of mutations in the population", default= 20)
    parser.add_argument("--K", type=int, help = "Carrying capacity", default = 10000)
    parser.add_argument("--time", type=int, help = "Time period of simulation", default = 100000)
    parser.add_argument("--tau", type=float, help = "Time step size for tau leaping algorithm", default = 2)
    parser.add_argument("--r", type=float, help="Base growth rate", default = 1)
    parser.add_argument("--mut_rate", type=float, help = "Mutation rate per genome per generation", default = 1e-7)
    parser.add_argument("--gamma", type = float, help ="Strength of NFDS", default = 10)
    parser.add_argument("--m", type=float, help = "Benefit due to mutation", default = 0.01)
    parser.add_argument("--sigma", type = float, help = "Recombination rate", default=1e-12) #recombination constant
    parser.add_argument("--delta", type = float, help = "Death rate", default = 0.1)
    parser.add_argument("--iter", type = int, help = "Repeat number", default = 0)
    parser.add_argument("--random_thres", type = int, help = "Threshold number of individuals below which randomization is applied", default = 100)
   
    #assign arguments to variables
    args = parser.parse_args()
    num_serotypes = args.num_serotypes
    num_mutations = args.init_num_mutations
    K = args.K
    time_end = args.time
    tau = args.tau
    r = args.r
    mut_rate = args.mut_rate
    gamma = args.gamma
    m = args.m
    sigma = args.sigma
    delta = args.delta
    iter = args.iter
    thres = args.random_thres
    numbers_init = np.zeros((num_serotypes, 2 ** num_mutations))
    numbers_init[:, 0] = (int)(K/(2*num_serotypes)) #sets WT strains numbers to 100
    output_filename = 'Output/mut_rate_'+str(mut_rate)+'_numm_'+str(num_mutations)+'_sero_'+str(num_serotypes)+'_g_'+str(gamma)+'_sig_'+str(sigma)+'_m_'+str(m)+'_iter_'+str(iter)+'.csv'
    #check if directory for output files exists and if it doesn't, creates it
    if (os.path.exists('Output')==False):
        os.mkdir('Output')

    #run model
    tosave, counter_overext, counter_mutsame = tau_leaping(num_serotypes, num_mutations, K, time_end, tau, r, mut_rate, gamma, m, sigma, delta, numbers_init, thres, output_filename)
    #save run to csv
    #df = tosave.loc[:, (tosave != 0).any(axis=0)]
    #df.to_csv('Output/mut_rate_'+str(mut_rate)+'_numm_'+str(num_mutations)+'_sero_'+str(num_serotypes)+'_g_'+str(gamma)+'_sig_'+str(sigma)+'_m_'+str(m)+'_iter_'+str(iter)+'.csv', sep = ',', index=False)
    print(counter_mutsame)
    #end = time.time()
    #print(end-start)

if __name__ == "__main__":
    main()