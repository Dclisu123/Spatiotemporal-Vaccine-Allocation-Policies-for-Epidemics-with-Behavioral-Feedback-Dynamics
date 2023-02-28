# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 13:37:51 2022

@author: dclisu
"""

import numpy as np
from simulation_heuristics_policy import simulate_policy_no_feedback
import matplotlib.pyplot as plt
import pandas as pd


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


#Data: Population and jurisdictions
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
comp_para[7,:] = np.asarray(np.asarray([1/3,1/3.2,1/3.4,1/3.6,1/3.8]))
comp_para[8,:] = 10**(-5) # Portion of initial infected
comp_para[9,:] = comp_para[7,:]*(comp_para[6,:]/comp_para[4,:] + comp_para[6,:]*comp_para[1,:]/comp_para[5,:] + (1-comp_para[1,:])/comp_para[5,:]) #R0

#Data: Feedback system
feedback_para = np.zeros((3,num_jur)) 
feedback_para[0,:] = np.linspace(15,35,num_jur) #lam_I
feedback_para[1,:] = 2*1/comp_para[2,:]*feedback_para[0,:] #lam_D
T_memory = 30 #Memory period
feedback_para[2,:] = np.ones((num_jur))*T_memory  #Memory

#Data: vaccination
#vacc_cap = np.sum(pop_vec) * 0.001


#Data: time horizon
tau = 250 #Days
#t_first_inf = np.ones((num_jur))*1
t_vacc = 75

total_policy = 9
fairness = np.zeros((total_policy))
newly_infections = np.zeros(9)
policy_list = ['No policy','Pro_rata','Incidence_rate','Mypoic','R0_eff','Final_size',\
               'R0_greater_1_pro_rate','R0_greater_1_R0_eff','Susp']

vacc_level = [0.001,0.002,0.003,0.005,0.01]
effi = np.zeros((len(vacc_level),total_policy))
fairness_V = np.zeros((len(vacc_level),total_policy))
fairness_I = np.zeros((len(vacc_level),total_policy))

for t in range(len(vacc_level)):
    vacc_cap = np.sum(pop_vec) * vacc_level[t]
    for i in range(total_policy):
        comp,new_I,new_D,x,fairness = simulate_policy_no_feedback(pop_vec,comp_para,vacc_cap,tau,t_vacc,i)
        newly_infections[i] = np.sum(new_I)
        effi[t,i] = np.sum(comp[:,t_vacc+1:,4])
        fairness_V[t,i] = gini(comp[:,tau-1,7])
        print(np.sum(comp[:,t_vacc+1:,4],axis = 1))
        fairness_I[t,i] = gini(np.sum(comp[:,t_vacc+1:,4],axis = 1))
        
        fig,ax = plt.subplots(nrows = 2,ncols =1 ,figsize=(20,8))
        plt.subplots_adjust(hspace=0.3)
        
        labels = ['J1','J2','J3','J4','J5']
        #ax = fig.add_subplot(2,1,1)
        for j in range(num_jur):
            ax[0].plot(np.arange(tau),new_I[j,:],label='J'+str(j+1),linewidth=3)
        ax[0].xaxis.set_major_locator(plt.MaxNLocator(6))
        ax[0].yaxis.set_major_locator(plt.MaxNLocator(6))
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].spines['bottom'].set_visible(False)
        ax[0].spines['left'].set_visible(False)
        ax[0].tick_params(labelsize=16) 
        ax[0].set_xlabel ('Time',fontsize=20)
        ax[0].set_ylabel ('New infections',fontsize=20)


        #ax = fig.add_subplot(2,1,2)
        for j in range(num_jur):
            ax[1].plot(np.arange(tau),x[j,:],label='J'+str(j+1),linewidth=3)
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
        plt.savefig('policy_'+policy_list[i]+'_no_behavior_'+str(t)+'.pdf')
        plt.show()
        
    effi[t,:] = effi[t,:]/effi[t,1]
    
df = pd.DataFrame(np.asarray([effi,fairness_V,fairness_I]).reshape(15,9))
df.to_excel(excel_writer = 'no_feed_back.xlsx')
