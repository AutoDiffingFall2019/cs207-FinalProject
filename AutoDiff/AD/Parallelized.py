#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 01:08:10 2019

"""
import numpy as np
from DualNumber_with_Reverse import DualNumber
import ElementaryFunctions_with_reverse as EF
class Parallelized_AD:
    '''
    parallelized AD class, 
    it stores a function or a list of functions
    each function is a string (Analogy to Matlab function handles)
    a list of variables required to calculate the derivative
    each variable is a string (Analogy to Matlab function handles)
    The main purpose is to return the Jacobian
    ref:https://www.mathworks.com/help/matlab/function-handles.html
    ******For pratical reason, each variable must add an _ in expression******
    For instance, 'sin(x1)' must be written as 'sin(_x1)' and variable name is 'x1'
    '''
    def __init__(self,fun=None,var=None):
        if fun:
            assert isinstance(fun,(str,list))
            self.function=fun if isinstance(fun,list) else [fun]
        else:
            self.function=[]
        if var:
            assert isinstance(var,(str,list))
            self.varname=var if isinstance(var,list) else [var]
        else:
            self.varname=[]
        self.variable=[]
        self.Jacobian=None
    def add_function(self,fun=None):
        if fun:
            assert isinstance(fun,str)
            self.function=[self.function] if isinstance(self.function,str) else self.function
            self.function.append(fun)
    def specify_variables(self,Var=None):
        if Var:
            assert isinstance(Var,(str,list))
            self.variable=[Var] if isinstance(Var,str) else Var
            
    def get_Jacobian(self,loc,forward=False):
        assert len(loc)==len(self.varname)
        self.Jacobian=np.zeros((len(self.function),len(self.varname)))
        for i, fun in enumerate(self.function):
            if forward:#Working based on Forward Mode
                translated_fun=self.Preprocess(fun)
                for j in range(len(self.varname)):
                    self.variable=[DualNumber(value,dual=0) for value in loc]   
                    self.variable[j]=DualNumber(loc[j],dual=1) 
                    element=eval(translated_fun)
                    self.Jacobian[i,j]=element.der
            else:#Working based on Backward Mode
                self.variable=[DualNumber(value,Reverse=True) for value in loc]        
                translated_fun=self.Preprocess(fun)
                element=eval(translated_fun)
                element.set_der(1)
                for j in range(len(self.varname)):
                    self.Jacobian[i,j]=self.variable[j].der
        return self.Jacobian
    
    def Preprocess(self,string:str):
        dictionary={'exp(':'EF.Exp(',
                    'sin(':'EF.Sin(',
                    'cos(':'EF.Cos(',
                    'tan(':'EF.Tan(',
                    'log(':'EF.Log(',
                    'arcsin(':'EF.ArcSin(',
                    'arccos(':'EF.ArcCos(',
                    'arctan(':'EF.ArcTan(',
                    'sqrt(':'EF.Sqrt(',
                    'power(':'EF.Power('}
        for key,item in dictionary.items():
            string=string.replace(key,item)    
        for i,name in enumerate(self.varname):
            string=string.replace('_'+name,f'self.variable[{i}]')

        #print(string) #For debugging
        return string
    
if __name__=='__main__':
    func=['_x**_y + sin(_x)',
          'exp(sin(_x+_y)+_z)-_x -sqrt(_y)', '_z*_x']
    #print('Functions:\nx +y +sin(x)\nexp(x+y)-x -sqrt(y)\nJacobian at [x,y]=[1,2]:')
    var_names=['x','y', 'z']
    PAD=Parallelized_AD(fun=func,var=var_names)
    print(PAD.get_Jacobian([1,2,3]))
    print('Working on Forward Mode:')
    print(PAD.get_Jacobian([1,2,3],forward=True))
    print('Adding another Functions:\nx +y +1,\nJacobian at [x,y]=[1,2]:')
    PAD.add_function('_x+1+_y ')
    print(PAD.get_Jacobian([1,2,3]))
