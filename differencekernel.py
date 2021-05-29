# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 09:47:33 2021

@author: kupij
"""
import numpy as np
import time
from scipy import linalg as la
from datetime import date



pot      =4
const    =1
reg      =0

traject_length = 1000
Timeinterval= 3
upbound     = 1
lowbound    = -1
dim         = 20
num_samples = 400
step_time = 1e-3  # time step size
nu = 1 # diffusion coefficient
lambd = 0.1 # cost parameter
gamma = 0 # discount factor
tol = 1e-13 #für while schleife
maxpolicy   = 50 #maximale Durchlaufzahl der while schleife

#Dateiname der Datei, in der die Parameter der Approximation der Valuefunction abgespeichert wird
savename = 'schl3000_pot'+str(pot)+'_const'+str(const)+'_dim'+str(dim)+'_nums'+str(num_samples)+'_reg'\
            +str(reg)+'_'+str(date.today())+'_maxpol'+str(maxpolicy)+'_tol'+str(tol)+'.npz'

grid = np.linspace(lowbound,upbound,dim)

step_spat = (upbound-lowbound)/(dim-1)

A = -2*np.diag(np.ones(dim), 0) + np.diag(np.ones(dim-1), 1) + np.diag(np.ones(dim-1), -1)
A[0,1] = 2; A[dim-1, dim-2] = 2
A = nu / step_spat**2 * A
B = (np.bitwise_and(grid > -0.4, grid < 0.4))*1
B_are = np.zeros((dim,1))
B_are[:,0] = B

R       = lambd * np.eye(1)
R_inv   = la.inv(R)

Q       = step_spat*np.eye(dim)
Q[0,0] /=2; Q[dim-1,dim-1] /=2

Q_t = step_time *Q
R_t = step_time *R

P_lin = la.solve_continuous_are(A, B_are, Q, R)

startwert1  = 1.2*np.ones(dim)
startwert2  = 2*(grid+1)**2*(grid - 1)**2

def rk4(state,control,rhs):
    k1      = step_time * rhs(state,control)
    k2      = step_time * rhs(state + 0.5*k1,control)
    k3      = step_time * rhs(state + 0.5*k2,control)
    k4      = step_time * rhs(state + 1.0*k3,control)
    return state + (k1+2*k2+2*k3+k4)/6

def rk4_direction(state,control,rhs):
    k1      = step_time * rhs(state,control)
    k2      = step_time * rhs(state + 0.5*k1,control)
    k3      = step_time * rhs(state + 0.5*k2,control)
    k4      = step_time * rhs(state + 1.0*k3,control)
    return (k1+2*k2+2*k3+k4)/6

def kernel(x1,y1):
    return np.power(x1.dot(y1)+const,pot)

# def kernel_diff(x1,y1,x2,y2):
#     pos = kernel(x1,x2) +kernel(y1,y2)
#     neg = kernel(y1,x2) + kernel(x1,y2)
#     return pos - neg

def eval_kernel(state,weights,initial_list,transported_list):
    eval_kernel = 0
    for i in range(len(initial_list)):
        eval_kernel += weights[i] * ( np.power(initial_list[i].dot(state)+const,pot))
    return eval_kernel

# def eval_kernel_diff(state,weights,initial_list,transported_list):
#     eval_kernel = 0
#     for i in range(len(initial_list)):
#         eval_kernel += weights[i] * ( np.power(initial_list[i].dot(state)+const,pot) - np.power(transported_list[i].dot(state)+const,pot))
#     return eval_kernel

def eval_surr_grad(samples_init,weights, state):
     grad = np.zeros(np.shape(state))
     for i in range(num_samples):
         grad += pot*weights[i]*( np.power(samples_init[i].dot(state)+1,pot-1)*samples_init[i])    
     return grad
 
# def eval_surr_grad(samples_init,list_transported,weights, state):
#      grad = np.zeros(np.shape(state))

#      for i in range(num_samples):
#          grad += pot*weights[i]*( np.power(samples_init[i].dot(state)+1,pot-1)*samples_init[i] \
#                                  -np.power(list_transported[i].dot(state)+1,pot-1)*list_transported[i] )
    
#      return grad

def rhs_schl(state,control):
    return A@state + state**3 + B*control

def rhs_lin(state,control):
    return A@state + B*control

def step_riccati_lin(state):
    control = (-R_inv @ B_are.T@P_lin@state)
    newstate = rk4(state,control,rhs_schl)
    return newstate,control

def step_pol(samples_init,weights,rhs,state):
    grad    = eval_surr_grad(samples_init,weights,state)
    control = -0.5*R_inv@B_are.T@grad
    newstate= rk4(state,control,rhs)
    return newstate, control

def running_cost(state,control):
    return state@Q_t@state +control@R_t@control

def create_transported_list_lin(initial_samples):
    transported_list = [np.zeros(dim)]
    run_cost = [0]*(len(initial_samples))
    for i in range(1,len(initial_samples)):
        state = initial_samples[i]
        run_cost[i] = (0.5*state@Q_t@state)
        for j in range(traject_length):
            state , control = step_riccati_lin(state)
            run_cost[i]+= running_cost(state,control)
            
        run_cost[i] -=(0.5*state@Q_t@state)
        transported_list.append(state)
    
    return transported_list, run_cost

def create_transported_list_pol(samples_init,weights,rhs):
    # print('Starte create_transported_list_pol\n')
    transported_list = [np.zeros(dim)]
    run_cost = [0]*len(initial_samples)
    
    for i in range(1,len(samples_init)):
        state = samples_init[i]
        run_cost[i] = (0.5*state@Q_t@state)
        
        for j in range(traject_length):
                state,control = step_pol(samples_init,weights, rhs,state)
                run_cost[i]+= running_cost(state,control)
        run_cost[i] -=(0.5*state@Q_t@state)
        transported_list.append(state)
        
    
    return transported_list, run_cost

def create_samples(size,dimension):
    list1 = [np.zeros(dim)]
    for i in range(size):
            list1.append(np.random.uniform(-1,1,dim))
    return list1

def create_kernelmatrix(list_initial):
    kernelmatrix = np.zeros((num_samples+1,num_samples+1))
    for i in range(num_samples+1):
        kernelmatrix[i,i] = kernel(list_initial[i],list_initial[i])
        for j in range(i+1,num_samples+1):
            kernelmatrix[i,j] = kernel(list_initial[i],list_initial[j])
            kernelmatrix[j,i] = kernelmatrix[i,j]
    return kernelmatrix

def test_surr(initial_list, weights,testsamplesize):
    testevals = np.zeros((testsamplesize,2))
    testset   = create_samples(testsamplesize,np.size(initial_list[0]))
    for i in range(testsamplesize):
        testevals[i][0] = eval_kernel(testset[i],weights, initial_list)
        testevals[i][1] = testset[i]@P_lin@testset[i]
    return testevals,testset

def comp_error(trans_old,trans_new):
    error_list = []
    for i in range(len(trans_old)):
        error_list.append(max(abs(trans_old[i]-trans_new[i])))
    error_max = max(error_list)
    return error_max

def policy_iteration(samp_init, transp_init, cost_init_list,rhs):
    print('Starte Policy Iteration \n')
    trans_pol = transp_init
    run_cost_list  = cost_init_list
    Kernelmatrix = create_kernelmatrix(samp_init)
    wght = np.linalg.solve(Kernelmatrix + reg*np.eye(np.shape(Kernelmatrix)[0]),run_cost_list)
    error_max = 1
    iteration = 0
    while error_max >tol and  iteration < maxpolicy:
        trans_pol_old = trans_pol
        #if iteration%5== 0:
        #    print(iteration,'Iterationen')
        # cost_compare(samp_init, wght, trans_pol, startwert1)
        trans_pol,run_cost_list = create_transported_list_pol(samp_init,wght,rhs)
        wght = np.linalg.solve(Kernelmatrix + reg*np.eye(np.shape(Kernelmatrix)[0]),run_cost_list)
        iteration +=1

        error_max = comp_error(trans_pol_old,trans_pol)
        print('\n error:\t', error_max,'\t',iteration)
    return(trans_pol, wght,run_cost_list)

def cost_compare(initial_samples,weights_op,x0,rhs):
    state_pol = x0
    kernelcost = (0.5*x0@Q_t@x0)
    # riccaticost= (0.5*x0@Q_t@x0)
    for j in range(int(Timeinterval/step_time)):
        # state_ric, control_ric  = step_riccati_lin(state_ric)
        # riccaticost            += running_cost(state_ric,control_ric)
        
        state_pol, control_pol  = step_pol(initial_samples,weights_op,rhs,state_pol)
        # print('\n', control_pol)
        # print('\n control\t',control_pol)
        kernelcost             += running_cost(state_pol,control_pol)
    kernelcost -= 0.5*state_pol@Q_t@state_pol
    print(kernelcost,'Kernelkosten\t\t\t\t final state', state_pol)
    return kernelcost,state_pol #riccaticost,state_ric

def cost_riccati(x0):
    x_iter = x0
    riccaticost= (0.5*x0@Q_t@x0)
    while np.linalg.norm(x_iter) > 1e-4:
        x_iter, control_ric     = step_riccati_lin(x_iter)
        riccaticost            += running_cost(x_iter,control_ric)
    return x_iter, riccaticost

'''
Die Bool-Abfragen dienen als Schalter um die einzelnen Aktionen auszufuehren oder nicht.
Auf if-else wird dabei verzichtet, weil beim Laden der Datei durch andere Programme 
alle Befehle der Datei ausgefuehrt werden. So werden zB. Plots in plotter.py erstellt und
es werden Funktionen aus dieser Datei benoetigt.
'''

if True == True:       #Neue Startwerte?
    initial_samples = create_samples(num_samples, dim)
    transported_list, run_cost = create_transported_list_lin(initial_samples)
    K = create_kernelmatrix(initial_samples)
    np.savez('differencekernel.npz', initial_samples = initial_samples, transported_list = transported_list, run_cost = run_cost)
    weights = np.linalg.solve(K+reg*np.eye(num_samples+1),run_cost)

if True == False:        #Startwerte laden?
    initial_samples = np.load('differencekernel.npz')['initial_samples']
    transported_list = np.load('differencekernel.npz')['transported_list']
    run_cost        =np.load('differencekernel.npz')['run_cost']
    K = create_kernelmatrix(initial_samples)
    weights = np.linalg.solve(K+reg*np.eye(num_samples+1),run_cost)

if True == True: #Policy Iteration durchfuehren?
    start = time.time()
    transported_list_pol,weights_op, run_cost_pol = policy_iteration(initial_samples,transported_list,run_cost,rhs_schl)
    end   = time.time()
    tt = end - start
    np.savez(savename, samp_init = initial_samples,transp = transported_list_pol, weights = weights_op)


if True == True: #Kosten vergleichen?
    kernelcost,fk  = cost_compare(initial_samples, weights_op,startwert1,rhs_schl) #riccaticost, finalstate_ric
if True == True: #Daten speichern?
    with open('cost_list.txt','a') as f:
        f.write('\n'+savename +'\t cost: '+str(kernelcost)+'\t fin_state: '\
                +str(np.round(fk/max(abs(fk)),2)) +'*'+str(max(abs(fk))) +'\t'+str(np.round(tt/3600,2)))
else:
    pass