# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 20:49:32 2023

@author: dclisu
"""
import numpy as np
from simulation_optimized import simulate,gini
from simulation_heuristics_policy import simulate_policy
import matplotlib.pyplot as plt
import pandas as pd




def plot(sol_x,T):
    total_policy = 9
    newly_infections = np.zeros((total_policy + len(sol_x)))
    fairness_v = np.zeros((total_policy + len(sol_x)))
    fairness_I = np.zeros((total_policy + len(sol_x)))
    ab_diff = np.zeros((total_policy + len(sol_x)))
    effi = np.zeros((total_policy + len(sol_x)))
    effi_days = np.zeros((total_policy + len(sol_x)))
    
    policy_list = ['No policy','Pro_rata','Incidence_rate','Mypoic','R0_eff','Final_size',\
                   'R0_greater_1_pro_rate','R0_greater_1_R0_eff','Susp']
     
    
    for i in range(len(sol_x)):
        comp,beta,new_I,new_D,diff_I,diff_D,x,fairness,fairness_i = simulate(pop_vec,comp_para,feedback_para,vacc_cap,tau,t_vacc,sol_x[i])
        x[:,tau-1] = 0
        newly_infections[i] = np.sum(new_I)
        fairness_v[i] = gini(comp[:,tau-1,7])
        fairness_I[i] = gini(np.sum(comp[:,t_vacc+1:,4],axis = 1))
        for j in range(num_jur):
            for k in range(j+1,num_jur):
                ab_diff[i] = ab_diff[i] + np.abs(comp[j,-1,7] - comp[k,-1,7]  )
        
        #effi[i] = np.sum(comp[:,tau-1,5] - comp[:,t_vacc,5] + comp[:,tau-1,6] - comp[:,t_vacc,6])
        effi_days[i] = np.sum(comp[:,t_vacc+1:,4])
        print(effi_days)
        
        fig,ax = plt.subplots(nrows=2,ncols =1 ,figsize=(20,8))
        plt.subplots_adjust(hspace=0.3)
        
        labels = ['J1','J2','J3','J4','J5']
        
        
        for j in range(num_jur):
            ax[0].plot(np.arange(tau),new_I[j,:],linewidth = 3)
        ax[0].xaxis.set_major_locator(plt.MaxNLocator(6))
        ax[0].yaxis.set_major_locator(plt.MaxNLocator(6))
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['bottom'].set_visible(False)
        ax[0].spines['left'].set_visible(False)
        ax[0].tick_params(labelsize=16) 
        ax[0].set_xlabel ('Time',fontsize=20)
        ax[0].set_ylabel ('New infections',fontsize=20)
        
        print('Optimal_obj:',np.sum(new_I))
        
        for j in range(num_jur):
            ax[1].plot(np.arange(tau),x[j,:],linewidth = 3)
        ax[1].xaxis.set_major_locator(plt.MaxNLocator(6))
        ax[1].yaxis.set_major_locator(plt.MaxNLocator(6))
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        ax[1].spines['left'].set_visible(False)
        ax[1].tick_params(labelsize=16) 
        ax[1].set_xlabel ('Time',fontsize=20)
        ax[1].set_ylabel ('Vaccinations',fontsize=20)
        
        
        
        fig.legend(labels, loc='right',fontsize = 25,borderaxespad=1.2)
        
        plt.plot()
        plt.savefig('policy_'+str(i)+'optimized_dynamics_'+str(T)+'_beta_'+str(vacc_level)+'.pdf')
        plt.show()
            

    for i in range(total_policy):
        comp,new_I,new_D,x,fairness,fairness_i = simulate_policy(pop_vec,comp_para,feedback_para,vacc_cap,tau,t_vacc,i)
        newly_infections[i+ len(sol_x)] = np.sum(new_I)
        fairness_v[i+ len(sol_x)] = gini(comp[:,tau-1,7])
        fairness_I[i+ len(sol_x)] = gini(np.sum(comp[:,t_vacc+1:,4],axis = 1))
        #effi[i+ len(sol_x)] = np.sum(comp[:,tau-1,5] - comp[:,t_vacc,5] + comp[:,tau-1,6] - comp[:,t_vacc,6])
        effi_days[i+ len(sol_x)] = np.sum(comp[:,t_vacc+1:,4])
        print(effi_days)
        print(policy_list[i]+'_obj:',np.sum(new_I))
        
        fig,ax = plt.subplots(nrows = 2,ncols =1 ,figsize=(20,8))
        plt.subplots_adjust(hspace=0.3)
        
        labels = ['J1','J2','J3','J4','J5']
        
        for j in range(num_jur):
            ax[0].plot(np.arange(tau),new_I[j,:],linewidth = 3)
        ax[0].xaxis.set_major_locator(plt.MaxNLocator(6))
        ax[0].yaxis.set_major_locator(plt.MaxNLocator(6))
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['bottom'].set_visible(False)
        ax[0].spines['left'].set_visible(False)
        ax[0].tick_params(labelsize=16) 
        ax[0].set_xlabel ('Time',fontsize=20)
        ax[0].set_ylabel ('New infections',fontsize=20)
        
        
        for j in range(num_jur):
            for k in range(j+1,num_jur):
                ab_diff[i+ len(sol_x)] = ab_diff[i+ len(sol_x)] + np.abs(comp[j,-1,7] - comp[k,-1,7]  )
        
        
        for j in range(num_jur):
            ax[1].plot(np.arange(tau),x[j,:],linewidth = 3)
        ax[1].xaxis.set_major_locator(plt.MaxNLocator(6))
        ax[1].yaxis.set_major_locator(plt.MaxNLocator(6))
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        ax[1].spines['left'].set_visible(False)
        ax[1].tick_params(labelsize=16) 
        ax[1].set_xlabel ('Time',fontsize=20)
        ax[1].set_ylabel ('Vaccinations',fontsize=20)
        
        fig.legend(labels, loc='right',fontsize = 25,borderaxespad=1.2)
        
        
        plt.plot()
        plt.savefig('policy_'+policy_list[i]+'_dynamics_'+str(T)+'_beta_'+str(vacc_level)+'.pdf')
        plt.show()
        
    effi = effi /effi[1 + len(sol_x)]
    effi_days = effi_days /effi_days[1 + len(sol_x)]
    return effi_days,np.asarray(fairness_v),np.asarray(fairness_I)










num_jur = 5
pop_size = 10**6
pop_vec = np.ones((num_jur))*pop_size

#Data: Compartmental related S,E,P,A,I,R,D,V
comp_para = np.zeros((10,num_jur))
comp_para[0,:] = np.ones((num_jur))*0.008 #infection fatality rate 
comp_para[1,:] = np.ones((num_jur))*0.15 # Portion of asymptomatic a
comp_para[2,:] = comp_para[0,:]/(1-comp_para[1,:]) #fatality rate p
comp_para[3,:] = np.ones((num_jur))*1/2.9 #rate from E to P, alpha
comp_para[4,:] = np.ones((num_jur))*1/2.3 #rate from P to (I+A), theta 
comp_para[5,:] = np.ones((num_jur))*1/10  #rate from (I+A) to (R+D), gamma
comp_para[6,:] = np.ones((num_jur))*0.25 # calibration of contact rate for P adn A, epsilon
#comp_para[7,:] = np.ones((num_jur))*1/2.5 # infection rate, beta
comp_para[7,:] = np.asarray([1/3.8,1/3.6,1/3.4,1/3.2,1/3])
comp_para[8,:] = 10**(-5) # Portion of initial infected
comp_para[9,:] = comp_para[7,:]*(comp_para[6,:]/comp_para[4,:] + comp_para[6,:]*comp_para[1,:]/comp_para[5,:] + (1-comp_para[1,:])/comp_para[5,:]) #R0

#Data: Feedback system
feedback_para = np.zeros((3,num_jur)) 
feedback_para[0,:] = np.linspace(15,35,num_jur) #lam_I
feedback_para[1,:] = 2*1/comp_para[2,:]*feedback_para[0,:] #lam_D
#T_memory = 60 #Memory period
#feedback_para[2,:] = np.ones((num_jur))*T_memory  #Memory

vacc_level = 0.002
#Data: vaccination
vacc_cap = np.sum(pop_vec) * vacc_level 
    

#Data: time horizon
tau = 600 #Days
#t_first_inf = np.ones((num_jur))*1
t_vacc = 200

T_list = [30,45,60,75]
total_data = np.zeros((len(T_list)*3,17))
for t in range(len(T_list)):
    T_memory = T_list[t] #Memory period
    feedback_para[2,:] = np.ones((num_jur))*T_memory  #Memory
    num_ini_start = 19
    results = np.load('../results/NLP_fixed_days_fairness_'+str(vacc_level)+'_'+str(T_list[t])+'.npz',allow_pickle= True)
    total_result = results['total_result']
    num_fixed_days = 8
    obj_x = np.zeros((num_fixed_days,num_ini_start))
    best_policy_x = np.zeros((num_fixed_days,num_jur,tau))
    best_obj = np.zeros((num_fixed_days))
    best_comp = np.zeros((num_fixed_days,num_jur,tau,8))
    absolute_diff = np.zeros((num_fixed_days))
    
    
    for i in range(num_fixed_days):
        for j in range(num_ini_start):
            cur_policy = total_result[i*num_ini_start + j,1]
            cur_comp,_,_,_,_,_,_,_,_ = simulate(pop_vec,comp_para,feedback_para,vacc_cap,tau,t_vacc,cur_policy)  
            obj_x[i,j] = np.sum(cur_comp[:,t_vacc+1:,4])
    best_idx = np.argmin(obj_x,axis = 1)
    
    for i in range(num_fixed_days):
        best_obj[i] = total_result[i*num_ini_start + best_idx[i],0]
        best_policy_x[i,:,:] = total_result[i*num_ini_start + best_idx[i],1]
        best_comp[i,:,:,:],_,_,_,_,_,_,_,_ = simulate(pop_vec,comp_para,feedback_para,vacc_cap,tau,t_vacc,best_policy_x[i,:,:]) 
        for j in range(num_jur):
            for k in range(j+1,num_jur):
                absolute_diff[i] = absolute_diff[i] + np.abs(best_comp[i,j,-1,7] -best_comp[i,k,-1,7])
    total_data[t,:],total_data[len(T_list)+t,:],total_data[2*len(T_list)+t,:] = plot(best_policy_x,T_list[t])

df = pd.DataFrame(total_data)
df.to_excel(excel_writer = 'result_vacc_' + str(vacc_level) +'.xlsx')

##plt.scatter(best_obj,best_fairness)
