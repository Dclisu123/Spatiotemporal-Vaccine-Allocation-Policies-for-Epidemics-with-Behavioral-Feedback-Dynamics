# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:16:43 2022

@author: dclisu
"""

import numpy as np

def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g   



def simulate(pop_vec,comp_para,feedback_para,vacc_cap,tau,t_vacc,x):
    #Initialize compartmental
    num_jur = len(pop_vec)
    comp = np.zeros((num_jur,tau,8)) #S,E,P,A,I,R,D,V
    comp[:,0,0] = 1-comp_para[8,:] # S is (1 - ini_infected) at start
    comp[:,0,4] = comp_para[8,:] # I is ini_infected 
    beta = np.zeros((num_jur,tau))
    a = comp_para[1,:] #a
    p = comp_para[2,:] #p
    alpha = comp_para[3,:] #alpha 
    theta = comp_para[4,:] #theta
    gamma = comp_para[5,:] #gamma
    epsilon = comp_para[6,:] #epsilon
    beta_base = comp_para[7,:] #basic transimission rate
    
    
    #Initialize feedback
    new_D = np.zeros((num_jur,tau))
    new_I = np.zeros((num_jur,tau))
    new_I[:,0] = comp_para[8,:]
    R0_basic = np.zeros((num_jur,tau))
    R0_eff = np.zeros((num_jur,tau))
    lam_I = feedback_para[0,:] #lam_I
    lam_D = feedback_para[1,:] #lam_D
    T = feedback_para[2,:] #Memory 
    M = np.zeros((num_jur,tau)) ## Percentage of M(t)---feedback factor
    u = np.zeros((num_jur,tau)) ## u = lam_I*new_I +lam_d*new_d
    diff_I = np.zeros((num_jur,tau))
    diff_D = np.zeros((num_jur,tau))
    
    
    
    for t in range(tau-1):
        #Update trasmission rate
        M[:,t] = 1/(1+np.exp(-u[:,t]))   
        beta[:,t] = beta_base * (1-M[:,t])
        R0_basic[:,t] = beta[:,t]*(epsilon/theta + epsilon*a/gamma+(1-a)/gamma)
        R0_eff[:,t] = R0_basic[:,t] * comp[:,t,0]
            
            
        #Update new Infected/Deceased for each time period
        new_I[:,t+1] = (1-a)*theta*comp[:,t,2]
        new_D[:,t+1] = p*gamma*comp[:,t,4]
        
                
        #Update Compartmental:S,E,P,A,I,R,D
        comp[:,t+1,0] = comp[:,t,0] - beta[:,t]*(comp[:,t,0]-x[:,t]/pop_vec)*(epsilon * (comp[:,t,1]+comp[:,t,3]) + comp[:,t,4]) - x[:,t]/pop_vec
        comp[:,t+1,1] = comp[:,t,1] + beta[:,t]*(comp[:,t,0]-x[:,t]/pop_vec)*(epsilon * (comp[:,t,1]+comp[:,t,3]) + comp[:,t,4]) - alpha * comp[:,t,1] 
        comp[:,t+1,2] = comp[:,t,2] + alpha * comp[:,t,1] - theta * comp[:,t,2] 
        comp[:,t+1,3] = comp[:,t,3] + a*theta*comp[:,t,2] - gamma*comp[:,t,3]
        comp[:,t+1,4] = comp[:,t,4] + new_I[:,t+1] - gamma*comp[:,t,4]
        comp[:,t+1,5] = comp[:,t,5] + gamma*(comp[:,t,3]+(1-p) * comp[:,t,4])
        comp[:,t+1,6] = comp[:,t,6] + new_D[:,t+1]
        comp[:,t+1,7] = comp[:,t,7] + x[:,t]/pop_vec
        
        
        
        #Feedback 
        for i in range(num_jur):
            #diff_I[i,t] = np.sum(new_I[i,max(0,t-int(T[i])+1):t+1]) - np.sum(new_I[i,max(0,t -2*int(T[i])+1) : max(0,t -int(T[i]) +1)]) 
            #diff_D[i,t] = np.sum(new_D[i,t-int(T[i])+1:t+1]) - np.sum(new_D[i,t -2*int(T[i]) + 1: t -int(T[i]) +1]) 
            for back_t in range(int(min(2*T[i],t+1))):
                if back_t < T[i]:
                    diff_I[i,t] += new_I[i,t-back_t]
                    diff_D[i,t] += new_D[i,t-back_t]
                else:
                    diff_I[i,t] -= new_I[i,t-back_t]
                    diff_D[i,t] -= new_D[i,t-back_t]
                #print(t-back_t)
        u[:,t+1] = lam_I * diff_I[:,t] + lam_D * diff_D[:,t]
             
            
            
    fairness = gini(comp[:,-1,7])      
    fairness_i = gini(np.sum(comp[:,:,4],axis = 1))

            
   # print(np.sum(comp,axis = 2))
            
    return comp,beta,new_I,new_D,diff_I,diff_D,x,fairness,fairness_i
            
            
            
            
            
            

            
            
            
        
    