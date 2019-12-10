#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 01:26:47 2019
Adams Optimization Algorithm based on
https://arxiv.org/abs/1412.6980
Implemented based on AutoDiffing library
CS207 Final Project
"""

from AD.DualNumber import DualNumber
import AD.ElementaryFunctions as EF
import numpy as np
class Adam_optimizer():
    def __init__(self,Objective_Grad=None,Target_Var=None):
        '''
        Objective_Grad: Gradient of Objective Function to be optimized
        Target_Var: Target variables need to be optimized
        '''
        self.Objective_Grad=Objective_Grad
        self.Target_Var=Target_Var
        self.set_parameters()
    def set_parameters(self,alpha=0.01,beta1=0.9,beta2=0.999,epsilon=1e-8,max_iter=1e5,tol=1e-8):
        '''
        Setting up hyderparameters used for the optimizer
        Adam has four hyperparameters:
            beta_1:float from 0 to 1, usually close to 1
            beta_2:float from 0 to 1, usually close to 1
            alpha :float, step size, usually close to zero
            epsilon :float, usually very close to zero
            max_iter :interger, max interations before force stop
            tol :float, tolerance before the optimzer should stop
        Detailed definition please refer to https://arxiv.org/pdf/1412.6980.pdf
        Default values borrowed from original paper, max iter set to a hundred thousand.
        '''
        self.alpha=alpha
        self.beta_1=beta1
        self.beta_2=beta2
        self.epsilon=epsilon
        self.max_iter=max_iter
        self.tol=tol

    def set_objective(self,Objective_Grad):
        '''
        Setting up objective function gradient
        input:
            Objective_Grad: function, A function returns the gradient of objective 
        
        '''
        self.Objective_Grad=Objective_Grad
#    def set_target_variables(self,Target_Var):
#        '''
#        Setting up objective function
#        input:
#            Target_Var: A list of variables, 
#        
#        '''
#        self.Target_Var=Target_Var
    def optimize(self,initialization=1):
        self._t=0 # Time =0
        self._m=0 # Initialize first momentum
        self._v=0# Initialize second momentum
        self._theta=initialization
        self.log=[]
        for i in np.arange(self.max_iter):
        	self._t+=1
        	g_t = self.Objective_Grad(self._theta)		#Get gradients w.r.t. stochastic objective at timestep t
        	self._m = self.beta_1*self._m + (1-self.beta_1)*g_t	#Update biased first moment estimate
        	self._v = self.beta_2*self._v + (1-self.beta_2)*(g_t*g_t)	#Update biased second raw moment estimate)
        	m_cap = self._m/(1-(self.beta_1**self._t))		#Compute bias-corrected first moment estimate
        	v_cap = self._v/(1-(self.beta_2**self._t))		#Compute bias-corrected second raw moment estimate
        	theta1 = self._theta - (self.alpha*m_cap)/(np.sqrt(v_cap)+self.epsilon)	#Update parameters
        	if abs( theta1-self._theta)< self.tol:		#checks convergence
        		break   
        	self._theta = theta1
        	self.log.append(theta1)
        return self._theta

if __name__=='__main__':
    def Demo_gradient(x):
        x=DualNumber(x);
        y=-EF.Sin(x)*EF.Cos(x)*EF.Tan(x)+EF.Exp(x)*EF.Log(x)*EF.Sqrt(x)*2
        return y.der #y=sin(x)cos(x)tan(x)-2exp(x)log(x)sqrt(x)
    Ad=Adam_optimizer(Demo_gradient)
    optimized_location=Ad.optimize()
    print('y=-sin(x)cos(x)tan(x)+2exp(x)log(x)sqrt(x) optimized starting from intialization 1 yields:\n',optimized_location)
    
    # import matplotlib.pyplot as plt
    # x= np.linspace(0.01, 1.2,1000)
    # y=-np.sin(x)*np.cos(x)*np.tan(x)+2*np.exp(x)*np.log(x)*np.sqrt(x)
    # plt.figure(dpi=500)
    # plt.plot(x,y,label='Objective')
    # x=Ad.log
    # y=-np.sin(x)*np.cos(x)*np.tan(x)+2*np.exp(x)*np.log(x)*np.sqrt(x)
    # plt.plot(x,y,'x',label='Trace')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.legend()
    # plt.title('Trajectory of Adam')
    
        