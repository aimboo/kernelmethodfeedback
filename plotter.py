# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 18:15:22 2021

@author: kupij
"""

from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la

def load_data(file,file2):
    '''
    load_data laedt die zum Plotten erforderten Daten, die in den npz files 
    abgespeichert wurden.
    '''
    global pot,const,samples,samples_t,weights,dim,trainset,weights_op
    
    pot     = int( file[ file.rfind('pot')+3:file.rfind('_const')]) #3 anstelle 5
    const   = float( file[ file.rfind('const')+5 : file.rfind('_dim')])
    dim     = int( file[ file.rfind('dim')+3: file.rfind('_nums')])
    
    samples_t   = np.load(file)['transp']
    samples     = np.load(file)['samp_init']
    weights     = np.load(file)['weights']
    
    trainset    = np.load(file2)['samp_init']
    weights_op  = np.load(file2)['weights']
#Die Sourcefiles muessen zuerst erzeugt werden mit den anderen Programmen
source      = ''
source_t    = ''
load_data(source,source_t)



T           = 5             #Zeitintervall
step_time   = 1e-3    # time step size
lambd       = 0.1 # cost parameter
steps       =int(T/step_time)



grid = np.linspace(-1,1,dim)

step_spat = 2/(dim-1)

A = -2*np.diag(np.ones(dim), 0) + np.diag(np.ones(dim-1), 1) + np.diag(np.ones(dim-1), -1)
A[0,1] = 2; A[dim-1, dim-2] = 2
A = 1/ step_spat**2 * A
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

x1  = 1.2*np.ones(dim)
x0  = 2*(grid+1)**2*(grid - 1)**2

def rhs_lin(state,control):
    return A@state + B*control

def rhs_schl(state,control):
    return A@state + state**3 + B*control

def rk4(state,control,rhs):
    k1      = step_time * rhs(state,control)
    k2      = step_time * rhs(state + 0.5*k1,control)
    k3      = step_time * rhs(state + 0.5*k2,control)
    k4      = step_time * rhs(state + 1.0*k3,control)
    return state + (k1+2*k2+2*k3+k4)/6

def running_cost(state,control):
    return state@Q_t@state +control@R_t@control

def eval_surr_grad(samples_init,list_transported,weights, state):
     grad = np.zeros(np.shape(state))
     for i in range(len(weights)):
         grad += pot*weights[i]*( np.power(samples_init[i].dot(state)+1,pot-1)*samples_init[i] \
                                 -np.power(list_transported[i].dot(state)+1,pot-1)*list_transported[i] )
     return grad

def eval_surr_grad_t(trainset,weights, state):
     grad = np.zeros(np.shape(state))
     for i in range(len(weights)):
         grad += pot*weights[i]*( np.power(trainset[i].dot(state)+1,pot-1)*trainset[i])    
     return grad

def step_pol(samples_init,list_transported,weights,rhs,state):
    grad    = eval_surr_grad(samples_init,list_transported,weights,state)
    control = -0.5*R_inv@B_are.T@grad
    newstate= rk4(state,control,rhs)
    return newstate, control

def step_pol_t(trainset,weights,rhs,state):
    grad    = eval_surr_grad_t(trainset,weights,state)
    control = -0.5*R_inv@B_are.T@grad
    newstate= rk4(state,control,rhs)
    return newstate, control

def step_ric(state,rhs):
    control = (-R_inv @ B_are.T@P_lin@state)
    newstate = rk4(state,control,rhs)
    return newstate,control

def create_plot_data(initial_samples,weights_op,transported_list_pol,x0,rhs):
    state_ric = x0
    state_pol = x0
    state_list = np.zeros((steps+1,dim))
    state_list[0] = x0
    state_list_r = np.zeros((steps+1,dim))
    state_list_r[0] = x0
    control_list = []
    control_list_r= []
    kernelcost   = (0.5*x0@Q_t@x0)
    kernelcost_r = (0.5*x0@Q_t@x0)
    for j in range(steps):
        state_pol, control_pol  = step_pol(initial_samples,transported_list_pol,weights_op,rhs,state_pol)
        state_ric,control_ric   = step_ric(state_ric,rhs)
        kernelcost             += running_cost(state_pol,control_pol)
        kernelcost_r           += running_cost(state_ric,control_ric)
        state_list[j+1] = state_pol
        state_list_r[j+1]= state_ric
        control_list.append(control_pol)
        control_list_r.append(control_ric)
    kernelcost -= 0.5*state_pol@Q_t@state_pol
    kernelcost_r -= 0.5*state_ric@Q_t@state_ric
    return kernelcost,state_list,control_list, kernelcost_r, state_list_r,control_list_r

def create_plot_data_t(trainset,weights_op,x0,rhs):
    state_pol = x0
    state_list = np.zeros((steps+1,dim))
    state_list[0] = x0
    control_list = []
    kernelcost   = (0.5*x0@Q_t@x0)
    for j in range(steps):
        state_pol, control_pol  = step_pol_t(trainset,weights_op,rhs,state_pol)
        kernelcost             += running_cost(state_pol,control_pol)
        state_list[j+1] = state_pol
        control_list.append(control_pol)
    kernelcost -= 0.5*state_pol@Q_t@state_pol
    return kernelcost,state_list,control_list


kernelcost, state_list, control_list, cost_r, states_r, control_r   = create_plot_data(samples, weights, samples_t, x0, rhs_schl)
kernelcost_t, state_list_t, control_list_t                          = create_plot_data_t(trainset, weights_op, x0, rhs_schl)
x = np.linspace(-1,1,dim)
t = np.linspace(0,T,steps+1)

X,T = np.meshgrid(x,t)



fig = plt.figure()
ax  = plt.axes(projection = '3d')

ax.plot_surface(X,T,state_list,cmap='plasma')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('y(x)')

plt.show()