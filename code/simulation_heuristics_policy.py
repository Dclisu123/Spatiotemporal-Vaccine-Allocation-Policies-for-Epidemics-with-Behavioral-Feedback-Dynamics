# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:16:43 2022

@author: dclisu
"""

import numpy as np
from scipy.optimize import root_scalar



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


def final_size(S_inf, S,E,P,A,I,beta,R_basic,gamma,epsilon):
    return -np.log(S_inf/S) + R_basic*(S_inf - S - E - P) - beta/gamma*(epsilon*A+I)



def simulate_policy(pop_vec,comp_para,feedback_para,vacc_cap,tau,t_vacc,vacc_policy):
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
    R0_basic = np.zeros((num_jur,tau))
    R0_eff = np.zeros((num_jur,tau))
    
    
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
    
    
    ##Vaccine policy
    x = np.zeros((num_jur,tau))
    
    for t in range(tau-1):
        #Update trasmission rate
        M[:,t] = 1/(1+np.exp(-u[:,t]))   
        beta[:,t] = beta_base * (1-M[:,t])
        R0_basic[:,t] = beta[:,t]*(epsilon/theta + epsilon*a/gamma+(1-a)/gamma)
        R0_eff[:,t] = R0_basic[:,t] * comp[:,t,0]
        
        
        
        #Vaccinations 
        if t >= t_vacc:
            if vacc_policy == 1: ##Pro_rata
                x[:,t] = np.amin(np.row_stack((vacc_cap*pop_vec/np.sum(pop_vec), comp[:,t,0]*pop_vec)),axis = 0)
            if vacc_policy == 2: #Incidence_rate
                inci_vec = pop_vec*beta[:,t]*(comp[:,t,0])*(epsilon * (comp[:,t,1]+comp[:,t,3]) + comp[:,t,4])
                if np.sum(inci_vec)==0:
                    x[:,t] = 0
                else:
                    x[:,t] = np.amin(np.row_stack((vacc_cap*inci_vec/np.sum(inci_vec),comp[:,t,0]*pop_vec)),axis =0)
            if vacc_policy ==3: #Myopic
                order = np.argsort(-beta[:,t]*(comp[:,t,0])*(epsilon * (comp[:,t,1]+comp[:,t,3]) + comp[:,t,4]))
                remain_vacc = vacc_cap
                item = 0
                while remain_vacc > 0 and item <num_jur:
                    x[order[item],t] = min(vacc_cap,comp[order[item],t,0]*pop_vec[order[item]])
                    remain_vacc -= x[order[item],t]
                    item +=1
            if vacc_policy ==4: #Effective R0
                R0_vec = pop_vec*R0_eff[:,t]
                if np.sum(R0_vec)==0:
                    x[:,t] = 0
                else:
                    x[:,t] = np.amin(np.row_stack((vacc_cap*R0_vec/np.sum(R0_vec),comp[:,t,0]*pop_vec)),axis =0)
            if vacc_policy ==5: #Final size
                S_inf = np.zeros((num_jur))
                for i in range(num_jur):
                    S_inf[i] = root_scalar(final_size,\
                                               args = (comp[i,t,0],comp[i,t,1],comp[i,t,2],comp[i,t,3],\
                                                       comp[i,t,4],beta[i,t],R0_basic[i,t],gamma[i],epsilon[i]),\
                                               method = 'brentq',bracket =[0,comp[i,t,0]] ,xtol = 0.000001 ).root
                if np.sum(pop_vec*(comp[:,t,0] - S_inf )) == 0:
                    x[:,t] =0
                else:
                    x[:,t] =np.amin(np.row_stack((vacc_cap*pop_vec*(comp[:,t,0] - S_inf )/np.sum(pop_vec*(comp[:,t,0] - S_inf)),comp[:,t,0]*pop_vec)),axis =0 )
            if vacc_policy == 6: #Pro_rata Effecitve R0>1 only
                pop_vec_larger = pop_vec*(R0_eff[:,t]>1)
                if np.sum(pop_vec_larger)==0:
                    x[:,t] = np.amin(np.row_stack((vacc_cap*pop_vec/np.sum(pop_vec), comp[:,t,0]*pop_vec)),axis = 0)
                else:
                    x[:,t] = np.amin(np.row_stack((vacc_cap*pop_vec_larger/np.sum(pop_vec_larger),comp[:,t,0]*pop_vec)),axis =0)  
            if vacc_policy ==7: #Effective R0, R0>1 only
                R0_vec_larger = pop_vec*(R0_eff[:,t]>1)*R0_eff[:,t]
                if np.sum(R0_vec_larger)==0:
                    if np.sum(R0_eff[:,t]) ==0:
                        x[:,t] = 0
                    else:
                        x[:,t] = np.amin(np.row_stack((vacc_cap*pop_vec*R0_eff[:,t]/np.sum(pop_vec*R0_eff[:,t]), comp[:,t,0]*pop_vec)),axis = 0)
                else:
                    x[:,t] = np.amin(np.row_stack((vacc_cap*R0_vec_larger/np.sum(R0_vec_larger),comp[:,t,0]*pop_vec)),axis =0)  
            if vacc_policy ==8: #based on S
                NS_vec = pop_vec*comp[:,t,0]
                if np.sum(NS_vec) ==0:
                    x[:,t]=0
                else:
                    x[:,t] = np.amin(np.row_stack((vacc_cap*NS_vec/np.sum(NS_vec),comp[:,t,0]*pop_vec)),axis =0)  
            
            
            
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

        # for i in range(num_jur):
        #     for t1 in range(min(t,T[j])):
        #         diff_I[i] += new_I[i,t-t1]
        #         diff_D[i] += new_I[i,t-t1]
        #     for t1 in range()

        
        
        for i in range(num_jur):
            #diff_I[i,t] = np.sum(new_I[i,max(0,t-int(T[i])+1):t+1]) - np.sum(new_I[i,max(0,t -2*int(T[i])+1) : max(0,t -int(T[i]) +1)]) 
            #diff_D[i,t] = np.sum(new_D[i,t-int(T[i])+1:t+1]) - np.sum(new_D[i,t -2*int(T[i]) + 1: t -int(T[i]) +1]) 
            for back_t in range(int(min(2*T[i],t+1))):
                if back_t < T[i]:
                    diff_I[i,t] += new_I[i,t-back_t]
                    diff_D[i,t] += new_D[i,t-back_t]
                    #print("add" + str(new_I[i,t-back_t]))
                else:
                    diff_I[i,t] -= new_I[i,t-back_t]
                    diff_D[i,t] -= new_D[i,t-back_t]
                    #print("minus" + str(new_I[i,t-back_t]))
        u[:,t+1] = lam_I * diff_I[:,t] + lam_D * diff_D[:,t]
            
            
    fairness = gini(comp[:,-1,7])
    fairness_i = gini(np.sum(comp[:,:,4],axis = 1))
            
   # print(np.sum(comp,axis = 2))
            
    return comp,new_I,new_D,x,fairness,fairness_i
            
            
            
            
            
            
def simulate_policy_no_feedback(pop_vec,comp_para,vacc_cap,tau,t_vacc,vacc_policy):
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
    R0_basic = np.zeros((num_jur,tau))
    R0_eff = np.zeros((num_jur,tau))
    
    
    #Initialize feedback
    new_D = np.zeros((num_jur,tau))
    new_I = np.zeros((num_jur,tau))
    new_I[:,0] = comp_para[8,:]
    R0_basic = np.zeros((num_jur,tau))
    R0_eff = np.zeros((num_jur,tau))
    #lam_I = feedback_para[0,:] #lam_I
    #lam_D = feedback_para[1,:] #lam_D
    #T = feedback_para[2,:] #Memory 
    #M = np.zeros((num_jur,tau)) ## Percentage of M(t)---feedback factor
    #u = np.zeros((num_jur,tau)) ## u = lam_I*new_I +lam_d*new_d
    #beta = np.ones((num_jur,tau))*beta_base
    
    ##Vaccine policy
    x = np.zeros((num_jur,tau))
    
    for t in range(tau-1):
        #Update trasmission rate
        #M[:,t] = 1/(1+np.exp(-u[:,t]))   
        beta[:,t] = beta_base * 1
        R0_basic[:,t] = beta[:,t]*(epsilon/theta + epsilon*a/gamma+(1-a)/gamma)
        R0_eff[:,t] = R0_basic[:,t] * comp[:,t,0]
        
        
        
        #Vaccinations 
        if t >= t_vacc:
            if vacc_policy == 1: ##Pro_rata
                x[:,t] = np.amin(np.row_stack((vacc_cap*pop_vec/np.sum(pop_vec), comp[:,t,0]*pop_vec)),axis = 0)
            if vacc_policy == 2: #Incidence_rate
                inci_vec = pop_vec*beta[:,t]*(comp[:,t,0])*(epsilon * (comp[:,t,1]+comp[:,t,3]) + comp[:,t,4])
                if np.sum(inci_vec)==0:
                    x[:,t] = 0
                else:
                    x[:,t] = np.amin(np.row_stack((vacc_cap*inci_vec/np.sum(inci_vec),comp[:,t,0]*pop_vec)),axis =0)
            if vacc_policy ==3: #Myopic
                order = np.argsort(-beta[:,t]*(comp[:,t,0])*(epsilon * (comp[:,t,1]+comp[:,t,3]) + comp[:,t,4]))
                remain_vacc = vacc_cap
                item = 0
                while remain_vacc > 0 and item <num_jur:
                    x[order[item],t] = min(vacc_cap,comp[order[item],t,0]*pop_vec[order[item]])
                    remain_vacc -= x[order[item],t]
                    item +=1
            if vacc_policy ==4: #Effective R0
                R0_vec = pop_vec*R0_eff[:,t]
                if np.sum(R0_vec)==0:
                    x[:,t] = 0
                else:
                    x[:,t] = np.amin(np.row_stack((vacc_cap*R0_vec/np.sum(R0_vec),comp[:,t,0]*pop_vec)),axis =0)
            if vacc_policy ==5: #Final size
                S_inf = np.zeros((num_jur))
                for i in range(num_jur):
                    S_inf[i] = root_scalar(final_size,\
                                               args = (comp[i,t,0],comp[i,t,1],comp[i,t,2],comp[i,t,3],\
                                                       comp[i,t,4],beta[i,t],R0_basic[i,t],gamma[i],epsilon[i]),\
                                               method = 'brentq',bracket =[0,comp[i,t,0]] ,xtol = 0.000001 ).root
                if np.sum(pop_vec*(comp[:,t,0] - S_inf )) == 0:
                    x[:,t] =0
                else:
                    x[:,t] =np.amin(np.row_stack((vacc_cap*pop_vec*(comp[:,t,0] - S_inf )/np.sum(pop_vec*(comp[:,t,0] - S_inf)),comp[:,t,0]*pop_vec)),axis =0 )
            if vacc_policy == 6: #Pro_rata Effecitve R0>1 only
                pop_vec_larger = pop_vec*(R0_eff[:,t]>1)
                if np.sum(pop_vec_larger)==0:
                    x[:,t] = np.amin(np.row_stack((vacc_cap*pop_vec/np.sum(pop_vec), comp[:,t,0]*pop_vec)),axis = 0)
                else:
                    x[:,t] = np.amin(np.row_stack((vacc_cap*pop_vec_larger/np.sum(pop_vec_larger),comp[:,t,0]*pop_vec)),axis =0)  
            if vacc_policy ==7: #Effective R0, R0>1 only
                R0_vec_larger = pop_vec*(R0_eff[:,t]>1)*R0_eff[:,t]
                if np.sum(R0_vec_larger)==0:
                    if np.sum(R0_eff[:,t]) ==0:
                        x[:,t] = 0
                    else:
                        x[:,t] = np.amin(np.row_stack((vacc_cap*pop_vec*R0_eff[:,t]/np.sum(pop_vec*R0_eff[:,t]), comp[:,t,0]*pop_vec)),axis = 0)
                else:
                    x[:,t] = np.amin(np.row_stack((vacc_cap*R0_vec_larger/np.sum(R0_vec_larger),comp[:,t,0]*pop_vec)),axis =0)  
            if vacc_policy ==8: #based on S
                NS_vec = pop_vec*comp[:,t,0]
                if np.sum(NS_vec) ==0:
                    x[:,t]=0
                else:
                    x[:,t] = np.amin(np.row_stack((vacc_cap*NS_vec/np.sum(NS_vec),comp[:,t,0]*pop_vec)),axis =0)  
            
            
            
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
        # diff_I = np.zeros((num_jur))
        # diff_D = np.zeros((num_jur))
        # for i in range(num_jur):
        #     for back_t in range(int(min(2*T[i],t))):
        #         if back_t < T[i]:
        #             diff_I[i] += new_I[i,t-back_t]
        #             diff_D[i] += new_D[i,t-back_t]
        #         else:
        #             diff_I[i] -= new_I[i,t-back_t]
        #             diff_D[i] -= new_D[i,t-back_t]
        # u[:,t+1] = lam_I * diff_I + lam_D * diff_D
             
            
            
    fairness = gini(comp[:,-1,7])

            
   # print(np.sum(comp,axis = 2))
            
    return comp,new_I,new_D,x,fairness
            
            
            
            
        
    