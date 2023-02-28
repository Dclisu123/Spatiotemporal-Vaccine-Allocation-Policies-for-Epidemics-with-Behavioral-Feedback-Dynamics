
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:04:40 2022

@author: dclisu
"""

import numpy as np
from gekko import GEKKO
from simulation_optimized import simulate
from simulation_heuristics_policy import simulate_policy
import concurrent.futures
import time 






def NLP_solve(para_list):
    
    ini_x,fixed_days,fairness_upper = para_list
    comp,hat_beta,new_I,new_D,_,_,_,_ = simulate(pop_vec,comp_para,feedback_para,vacc_cap,tau,t_vacc,ini_x)
    
    #Simulation
    #Optimization
    m = GEKKO(remote=False)
    var_x = m.Array(m.Var,(num_jur,tau-t_vacc),value= 0.0,lb=0)
    var_comp = m.Array(m.Var,(num_jur,tau-t_vacc,8),value= 0.0,lb = 0)
    var_beta = m.Array(m.Var,(num_jur,tau-t_vacc),value= 0.0,lb = 0)
    var_new_I = m.Array(m.Var,(num_jur,tau),value= 0.0,lb = 0)
    var_new_D = m.Array(m.Var,(num_jur,tau),value= 0.0,lb = 0)
    var_comp[:,0,:] =  comp[:,t_vacc,:]
    var_beta[:,0] = hat_beta[:,t_vacc]
    var_new_I[:,:t_vacc+1] = new_I[:,:t_vacc+1]
    var_new_D[:,:t_vacc+1] = new_D[:,:t_vacc+1]
    var_diff_I = m.Array(m.Var,(num_jur,tau-t_vacc),value= 0.0)
    var_diff_D = m.Array(m.Var,(num_jur,tau-t_vacc),value= 0.0)
    
    x_abs_diff = m.Var(0.0)
    
    for t in range((tau-t_vacc)):
        for j in range(num_jur):
            var_x[j,t].value = ini_x[j,t_vacc+t]
            
            
    for t in range(tau-t_vacc-1):
        period = int(t/fixed_days)
        fix = True
        if t%fixed_days == 0:
            fix = False
        for j in range(num_jur):
            var_new_I[j,t+t_vacc+1] = m.Intermediate((1-comp_para[1,j])*comp_para[4,j]*var_comp[j,t,2])
            var_new_D[j,t+t_vacc+1] = m.Intermediate(comp_para[2,j] *comp_para[5,j]*var_comp[j,t,4])
            
            
            var_comp[j,t+1,0] = m.Intermediate(var_comp[j,t,0] - var_beta[j,t]*(var_comp[j,t,0]-var_x[j,t]/pop_vec[j])*(comp_para[6,j]*(var_comp[j,t,1]+var_comp[j,t,3])+ var_comp[j,t,4]) - var_x[j,t]/pop_vec[j])
            var_comp[j,t+1,1] = m.Intermediate(var_comp[j,t,1] + var_beta[j,t]*(var_comp[j,t,0]-var_x[j,t]/pop_vec[j])*(comp_para[6,j]*(var_comp[j,t,1]+var_comp[j,t,3])+var_comp[j,t,4]) - comp_para[3,j] * var_comp[j,t,1]) 
            var_comp[j,t+1,2] = m.Intermediate(var_comp[j,t,2] + comp_para[3,j] * var_comp[j,t,1] - comp_para[4,j] * var_comp[j,t,2]) 
            var_comp[j,t+1,3] = m.Intermediate(var_comp[j,t,3] + comp_para[1,j]*comp_para[4,j]*var_comp[j,t,2] - comp_para[5,j]*var_comp[j,t,3])
            var_comp[j,t+1,4] = m.Intermediate(var_comp[j,t,4] + var_new_I[j,t+t_vacc+1] - comp_para[5,j]*var_comp[j,t,4])
            var_comp[j,t+1,5] = m.Intermediate(var_comp[j,t,5] + comp_para[5,j]*(var_comp[j,t,3]+(1-comp_para[2,j]) * var_comp[j,t,4]))
            var_comp[j,t+1,6] = m.Intermediate(var_comp[j,t,6] + var_new_D[j,t+t_vacc+1])
            var_comp[j,t+1,7] = m.Intermediate(var_comp[j,t,7] + var_x[j,t]/pop_vec[j])
            for q in range(8):
                m.Equation(var_comp[j,t+1,q]>=0)
                
            if fix == True:
                m.Equation(var_x[j,period*fixed_days] == var_x[j,t] )
                
                
                
            m.Obj(var_comp[j,t+1,4])
        m.Equation(m.sum(var_x[:,t]) <= vacc_cap)

        
        
        

        for j in range(num_jur):
            var_diff_I[j,t] = m.Intermediate( m.sum(var_new_I[j,t+t_vacc-int(feedback_para[2,j])+1: t + t_vacc + 1]) - m.sum(var_new_I[j,t + t_vacc-2*int(feedback_para[2,j]) + 1 : t + t_vacc -int(feedback_para[2,j])+1 ])  )
            var_diff_D[j,t] = m.Intermediate( m.sum(var_new_D[j,t+t_vacc-int(feedback_para[2,j])+1: t + t_vacc + 1]) - m.sum(var_new_D[j,t + t_vacc-2*int(feedback_para[2,j]) + 1 : t + t_vacc -int(feedback_para[2,j])+1 ])  )
            var_beta[j,t+1] = m.Intermediate(comp_para[7,j]*(1- 1/(1+m.exp(-(feedback_para[0,j]* var_diff_I[j,t] + feedback_para[1,j]*var_diff_D[j,t])))))
    
    # for i in range(num_jur):
    #     for j in range(i+1,num_jur):
    #         x_abs_diff = x_abs_diff + m.abs(var_comp[i,-1,7] - var_comp[j,-1,7])
            
   # m.Equation(x_abs_diff <= fairness_upper)
            
    sol_x = np.zeros((num_jur,tau))
    
    try:
         m.options.solver = 1
         m.solve()
         
         for t in range((tau-t_vacc)):
            for j in range(num_jur):
                sol_x[j,t+t_vacc] = var_x[j,t].value[0]
                
    except Exception:
        pass   
 
    comp,hat_beta,new_I,new_D,diff_I,diff_D,_,_ = simulate(pop_vec,comp_para,feedback_para,vacc_cap,tau,t_vacc,sol_x)
    obj_value = (np.sum(comp[:,t_vacc+1:,4]))
    #fairness_level = gini(np.sum(sol_x,axis = 1))


    return obj_value,sol_x


if __name__ == '__main__':
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
    comp_para[7,:] = np.ones((num_jur))*1/2.5 # infection rate, beta
    comp_para[8,:] = 10**(-5) # Portion of initial infected
    comp_para[9,:] = comp_para[7,:]*(comp_para[6,:]/comp_para[4,:] + comp_para[6,:]*comp_para[1,:]/comp_para[5,:] + (1-comp_para[1,:])/comp_para[5,:]) #R0
    
    #Data: Feedback system
    feedback_para = np.zeros((3,num_jur)) 
    feedback_para[0,:] = np.linspace(15,35,num_jur) #lam_I
    feedback_para[1,:] = 2*1/comp_para[2,:]*feedback_para[0,:] #lam_D
    T_memory = 60#Memory period
    feedback_para[2,:] = np.ones((num_jur))*T_memory  #Memory
    
    #Data: vaccination
    vacc_cap = np.sum(pop_vec) * 0.01
    
    
    #Data: time horizon
    tau = 600 #Days
    #t_first_inf = np.ones((num_jur))*1
    t_vacc = 200

    
    
    
    
    
    start_time = time.time()
    vacc_list = [0,1,2,3,4,5,6,7,8]
    ini_x_list = []
    result_policy = np.zeros((9))
    for vacc_policy in vacc_list:
        obj_policy,_,_,policy_x,_ = simulate_policy(pop_vec,comp_para,feedback_para,vacc_cap,tau,t_vacc,vacc_policy)
        ini_x_list.append(policy_x)
        result_policy[vacc_policy] = np.sum(obj_policy[:,t_vacc+1:,4])
    
    num_exp = 10
    for i in range(num_exp):
        cur_x = np.zeros((num_jur,tau))
        cur_x [:,t_vacc:]= np.random.rand(num_jur,tau-t_vacc)*vacc_cap
        ini_x_list.append(cur_x)
    
    
    fairness_upper = 0.2
    fix_days_list = [1,7,14,30,60,90,180,240]
    para_list = []
    for i in fix_days_list:
        for j in ini_x_list:
            para_list.append((j,i,fairness_upper))
    
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(NLP_solve,para_list)
    total_result = []
    for lines in results:
        total_result.append(lines)

    np.savez('NLP_fixed_days_fairness_0.01_60.npz', total_result = np.array(total_result,dtype=object))
    
    #obj_x,sol_x,fairness = NLP_solve(para_list[0])
    
    total_time = time.time() - start_time
    
    print('finish time:%.4f' %total_time)

        

