#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 01:08:10 2019

"""
import numpy as np
from DualNumber import DualNumber
import ElementaryFunctions as EF

class Parallelized_AD:
    """
    DESCRIPTION
    =======
    A class to run automatic differentiation in both forward and reverse mode for vector
    functions of vectors and scalars.  It stores a function or a list of functions, where
    each function is a string (similar to Matlab function handles).  A list of variables 
    is required to calculate the derivative.  For instance, 'sin(x1)' must be written as 
    func ='sin(_x1)' and var_names = 'x1'.  To initalize a Parallelized_AD instance, the user must then
    write PAD=Parallelized_AD(fun=func,var=var_names).
    
    INPUTS
    =======
    fun: array, default None
        Stores the list of functions for which we will eventually calculate the Jacobian. Note that all variables
        in the function array must be preceded by an underscore '_', e.g. _x.
    var: array, default None
        Stores list of variables (will differentiate function wrt each)
    Reverse: Boolean, optional, if set to true the gradient will be calculated based on reverse mode
        Default value is false, in which case forward mode is used.
    RETURNS
    ========
    Parallelized_AD: An instance with methods such as get_Jacobian() to get the value of the Jacobian
    at a certain point.  The user can also add functions to the current list of functions with the 
    add_function() method.
    
    
    NOTES
    =====
    PRE:
        - fun, var are arrays or strings
        - get_Jacobian() method takes numeric loc array as an input, the location where 
        the Jacobian should be evaluated
        
    POST:
        - function, variables will be stored in public attributes self.function, self.variable
        - self.Jacobian is initialized to None
        - raises a AttributeError exception if function input is not a list or string
        - raises a AttributeError exception if variable input is not a list or string
        - defines the overloading of arithmetic operators and unary operators
        
    EXAMPLES
    =========
    >>> Parallelized_AD(fun=['_x**_y + logit(_x+_y) + _z'],var=['x','y', 'z']).get_Jacobian([0.5,0.25,1])
    array([[ 0.6383432 , -0.36496999,  1.        ]])
    >>> func = ['_x + sin(_y)*_z', '_x + sin(_y)*exp(_z)']
    >>> PAD = Parallelized_AD(fun = func, var = ['x', 'y', 'z'])
    >>> PAD.get_Jacobian([1,2,3])
    array([[ 1.        , -1.24844051,  0.90929743],
           [ 1.        , -8.35853265, 18.26372704]])
    >>> func = ['_x + sin(_y)*_z', '_x + sin(_y)*exp(_z)']
    >>> PAD = Parallelized_AD(fun = func, var = ['x', 'y', 'z'])
    >>> PAD.add_function('_x+_y+1')
    >>> PAD.get_Jacobian([1,2,3], forward=True)
    array([[ 1.        , -1.24844051,  0.90929743],
           [ 1.        , -8.35853265, 18.26372704],
           [ 1.        ,  1.        ,  0.        ]])
    """

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
        self._Jacobian=None
        self._value = None
    
    @property
    def Jacobian(self):
        return self._Jacobian
    
    @property
    def value(self):
        return self._value
        
    def add_function(self,fun):
        """
        DESCRIPTION
        =======
        A class method to add a function to an existing Parallelized_AD object.  For example:
        >>> PAD = Parallelized_AD(fun=['_x**_y + sin(_x) + _z'],var=['x','y', 'z'])
        >>> PAD.add_function('_x')
        >>> print(PAD.function)
        ['_x**_y + sin(_x) + _z', '_x']
        
        Assert conditions ensure input is a string
        """
        assert isinstance(fun,str)
        self.function=[self.function] if isinstance(self.function,str) else self.function
        self.function.append(fun)
            
    def add_var(self,var):
        """
        DESCRIPTION
        =======
        A class method to add variables to an existing Parallelized_AD object.  For example:
        >>> PAD = Parallelized_AD(fun=['_x**_y + sin(_x)'],var=['x','y'])
        >>> PAD.add_function('_z')
        >>> PAD.add_var('z')
        >>> print(PAD.function)
        ['_x**_y + sin(_x)', '_z']
        >>> print(PAD.varname)
        ['x', 'y', 'z']
        
        Assert conditions ensure input is a string
        """
        assert isinstance(var,str)
        self.varname=[self.varname] if isinstance(self.varname,str) else self.varname
        self.varname.append(var)
            
    def get_Jacobian(self,loc,forward=False):
        """
        DESCRIPTION
        =======
        A class method to get the Jacobian of user-specified vector function.  See
        doctests for details on usage.  User inputs location to take Jacobian, and
        function returns the Jacobian at that point.  Assert conditions ensure 
        dimension of location input is the same as the number of variables.
        
        forward argument specifies whether reverse (False) or forward (True) mode
        is used
        """

        assert len(loc)==len(self.varname) 
        self._Jacobian=np.zeros((len(self.function),len(self.varname)))
        
        # for each function, if forward, do forward mode calculation, else do reverse
        # see documentation for details on reverse mode calculation
        for i, fun in enumerate(self.function):
            if forward:
                # pre-process each function to be differentatied
                translated_fun=self.preprocess(fun)
                # for each variable, take the derivative at the value specified
                for j in range(len(self.varname)):
                    self.variable=[DualNumber(value,dual=0) for value in loc]   
                    self.variable[j]=DualNumber(loc[j],dual=1) 
                    element=eval(translated_fun)
                    self._Jacobian[i,j]=element.der
            # similarly, if reverse, carry out reverse mode
            else:
                
                self.variable=[DualNumber(value,Reverse=True) for value in loc]
                translated_fun=self.preprocess(fun)
                element=eval(translated_fun)
                element.set_der(1)
                for j in range(len(self.varname)):
                    self._Jacobian[i,j]=self.variable[j].der
        return self._Jacobian
    
    def get_value(self,loc):
        """
        DESCRIPTION
        =======
        A class method to get the value of user-specified vector function.  See
        doctests for details on usage.  User inputs location of vector-valued function,
        and get_value returns the value at the specified location.
        
        >>> PAD = Parallelized_AD(fun=['_x * arcSin(_y*_z)+_x'], var=['x', 'y', 'z'])
        >>> print(PAD.get_value([0.4,0.2,1]))
        [0.48054317]
        >>> print(PAD.get_Jacobian([0.4,0.2,1]))
        [[1.20135792 0.40824829 0.08164966]]
        """
        assert len(loc)==len(self.varname)
        self._value=np.zeros((len(self.function)))
        
        # for each function, if forward, do forward mode calculation, else do reverse
        # see documentation for details on reverse mode calculation
        for i, fun in enumerate(self.function):
            
            # pre-process each function to be differentatied
            translated_fun=self.preprocess(fun)

            # for each variable, take the derivative at the value specified
            # for each variable, take the derivative at the value specified
            self.variable=[DualNumber(value) for value in loc]
            element=eval(translated_fun)
            self._value[i]=element.val
        return self._value
        
    
    def preprocess(self, string:str):
        """
        DESCRIPTION
        =======
        A class method to process the user input functions into a form such that we 
        can use our DualNumber class and ElementaryFunction module.
        
        >>> PAD = Parallelized_AD(fun=['sin(_x)'],var=['x'])
        >>> print(PAD.preprocess(PAD.function[0]))
        EF.Sin(self.variable[0])
        """
        dictionary={'sin(':'EF.Sin(',
                    'tan(':'EF.Tan(',
                    'cos(':'EF.Cos(',
                    'exp(':'EF.Exp(',
                    'power(':'EF.Power(',
                    'log(':'EF.Log(',
                    'arcSin(':'EF.ArcSin(',
                    'arcCos(':'EF.ArcCos(',
                    'arcTan(':'EF.ArcTan(',
                    'sqrt(':'EF.Sqrt(',
                    'sinh(':'EF.Sinh(',
                    'cosh(':'EF.Cosh(',
                    'tanh(':'EF.Tanh(',
                    'logit(':'EF.L('}
        for key,item in dictionary.items():
            string=string.replace(key,item)    
        for i,name in enumerate(self.varname):
            string=string.replace('_' + name,f'self.variable[{i}]')

        return string
   
if __name__=='__main__':   
    
    import doctest
    doctest.testmod(verbose=True)
    
