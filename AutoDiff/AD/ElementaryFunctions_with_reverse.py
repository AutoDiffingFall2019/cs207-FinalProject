#!/anaconda3/bin/python
# -*- coding: utf-8 -*-

from DualNumber_with_reverse import DualNumber
import numpy as np
import math


def Sin(x):
    '''
    >>> print(Sin(DualNumber(5,1)))
    Derivative: 0.28
    Value: -0.96
    '''
    if data_type_check(x) == 0:
        if x._rev:
            z=DualNumber(np.sin(x._val),Reverse=True)
            x.children.append((np.cos(x._val),z))
            return z
        return DualNumber(np.sin(x._val),np.cos(x._val)*x._der)
    else:
        return DualNumber(np.sin(x),0)

    
def Tan(x):
    '''
    >>> print(Tan(DualNumber(5,1)))
    Derivative: 12.43
    Value: -3.38
    '''
    if data_type_check(x) == 0:
        if x._rev:
            z=DualNumber(np.tan(x._val),Reverse=True)
            x.children.append(((1+np.tan(x._val)*np.tan(x._val)),z))
            return z
        return DualNumber(np.tan(x._val),(1+np.tan(x._val)*np.tan(x._val))*x._der)
    else:
        return DualNumber(np.tan(x),0)


def Cos(x):
    '''
    >>> print(Cos(DualNumber(5,1)))
    Derivative: 0.96
    Value: 0.28
    '''
    if data_type_check(x) == 0:
        if x._rev:
            z=DualNumber(np.cos(x._val),Reverse=True)
            x.children.append((-1*np.sin(x._val),z))
            return z
        return DualNumber(np.cos(x._val),-1*np.sin(x._val)*x._der)
    else:
        return DualNumber(np.cos(x),0)

def Exp(x):
    '''
    >>> print(Exp(DualNumber(5,1)))
    Derivative: 148.41
    Value: 148.41
    '''
    if data_type_check(x) == 0:
        if x._rev:
            z=DualNumber(np.exp(x._val),Reverse=True)
            x.children.append((np.exp(x._val),z))
            return z        
        return DualNumber(np.exp(x._val),np.exp(x._val)*x._der)
    else:
        return DualNumber(np.exp(x),0)

def Power(x,n):
    '''
    >>> print(Power(DualNumber(5,1),2))
    Derivative: 10.00
    Value: 25.00
    '''
    if data_type_check(x) == 0:
        if x._rev:
            z=DualNumber(x._val**n,Reverse=True)
            x.children.append(((n*(x._val**(n-1)),z)))
            return z
        return DualNumber(x._val**n,n*(x._val**(n-1))*x._der)
    else:
        return DualNumber(x**n,0)

def Log(x, base = np.exp(1)):
    '''
    >>> print(Log(DualNumber(5,1)))
    Derivative: 0.20
    Value: 1.61
    '''
    if data_type_check(x) == 0:
        if x._rev:
            z=DualNumber(np.log(x._val)/np.log(base), Reverse=True)
            x.children.append((1/(x._val*np.log(base)),z))
            return z          
        return DualNumber(np.log(x._val)/np.log(base), (1/(x._val*np.log(base)))*x._der)
    else:
        return DualNumber(np.log(x)/np.log(base), 0)

        
def ArcSin(x):
    '''
    >>> print(ArcSin(DualNumber(0.5)))
    Derivative: 1.15
    Value: 0.52
    '''
    if abs(x) >= math.pi:
        raise ValueError('ArcSin is only defined on (-pi,pi)!')
    if data_type_check(x) == 0:
        if x._rev:
            z=DualNumber(np.arcsin(x._val),Reverse=True)
            x.children.append((1/np.sqrt(1-x._val**2),z))
            return z          
        return DualNumber(np.arcsin(x._val),1/np.sqrt(1-x._val**2) * x._der)
    else:
        return DualNumber(np.arcsin(x),0)


def ArcCos(x):
    '''
    >>> print(ArcCos(DualNumber(0.5)))
    Derivative: -1.15
    Value: 1.05
    '''
    if abs(x) >= math.pi:
        raise ValueError('ArcCos is only defined on (-pi,pi)!')
    if data_type_check(x) == 0:
        if x._rev:
            z=DualNumber(np.arccos(x._val),Reverse=True)
            x.children.append((-1/np.sqrt(1-x._val**2),z))
            return z
        return DualNumber(np.arccos(x._val),-1/np.sqrt(1-x._val**2) * x._der)
    else:
        return DualNumber(np.arccos(x),0)
            
def ArcTan(x):
    '''
    >>> print(ArcTan(DualNumber(0.5)))
    Derivative: 0.80
    Value: 0.46
    '''
    if abs(x) >= 1:
        raise ValueError('ArcTan is only defined on (-1,1)!')
    if data_type_check(x) == 0:
        if x._rev:
            z=DualNumber(np.arctan(x._val),Reverse=True)
            x.children.append((1/(1+x._val**2),z))
            return z        
        return DualNumber(np.arctan(x._val),1/(1+x._val**2) * x._der)
    else:
        return DualNumber(np.arctan(x),0)

def Sqrt(x):
    '''
    >>> print(Sqrt(DualNumber(9,1)))
    Derivative: 0.17
    Value: 3.00
    >>> print(Sqrt(DualNumber(9,Reverse = True)))
    Derivative: 0.00
    Value: 3.00
    '''
    if data_type_check(x) == 0:
        if x._rev:
            z=DualNumber(np.sqrt(x._val),Reverse=True)
            x.children.append((1/(2*np.sqrt(x._val)),z))
            return z             
        return DualNumber(np.sqrt(x._val),1/(2*np.sqrt(x._val)) * x._der)
    else:
        return DualNumber(np.sqrt(x),0)

# Hyperbolic functions
def Sinh(x):
    '''
    >>> print(Sinh(DualNumber(1,1)))
    Derivative: 1.54
    Value: 1.18
    '''
    if data_type_check(x) == 0:
        if x._rev:
            z=DualNumber((np.exp(x._val)-np.exp(-x._val))/2,Reverse=True)
            x.children.append(((np.exp(x._val)+np.exp(-x._val))/2, z))
            return z
        return DualNumber((np.exp(x._val)-np.exp(-x._val))/2,((np.exp(x._val)+np.exp(-x._val))/2)*x._der)
    else:
        return DualNumber((np.exp(x)-np.exp(-x))/2,0)

def Cosh(x):
    '''
    >>> print(Cosh(DualNumber(1,Reverse =  True)))
    Derivative: 0.00
    Value: 1.54
    '''
    if data_type_check(x) == 0:
        if x._rev:
            z=DualNumber((np.exp(x._val)+np.exp(-x._val))/2,Reverse=True)
            x.children.append(((np.exp(x._val)-np.exp(-x._val))/2, z))
            return z
        return DualNumber((np.exp(x._val)+np.exp(-x._val))/2,((np.exp(x._val)-np.exp(-x._val))/2)*x._der)
    else:
        return DualNumber((np.exp(x)+np.exp(-x))/2,0)

def Tanh(x):
    '''
    >>> print(Tanh(DualNumber(0,1)))
    Derivative: 1.00
    Value: 0.00
    '''
    if data_type_check(x) == 0:
        Z = (np.exp(x._val)-np.exp(-x._val))/(np.exp(x._val)+np.exp(-x._val))
        if x._rev:
            z=DualNumber(Z,Reverse=True)
            x.children.append((1-Z**2, z))
            return z
        return DualNumber(Z,(1-Z**2)*x._der)
    else:
        return DualNumber((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)), 0)

# Logistic functions
def L(x):
    '''
    >>> print(L(DualNumber(0.5,Reverse = True)))
    Derivative: 0.00
    Value: 0.62
    '''
    if abs(x) >= 1:
        raise ValueError('Logit is only defined on (-1,1)!')
    if data_type_check(x) == 0:
        if x._rev:
            z=DualNumber(1/(1+np.exp(-x._val)),Reverse=True)
            x.children.append((np.exp(x._val)/(1+np.exp(x._val))**2, z))
            return z
        return DualNumber(1/(1+np.exp(-x._val)),(np.exp(x._val)/(1+np.exp(x._val))**2)*x._der)
    else:
        return DualNumber(1/(1+np.exp(-x)), 0)


def data_type_check(x):
   try:
       if x._der==None and x._rev==True:
           return 0  # returns 0 if x is DualNumber
       float(x._val)+float(x._der)
       return 0  # returns 0 if x is DualNumber
   except AttributeError:
       try:
           float(x)
           return 1 # returns 1 if x is real
       except:
           raise AttributeError('Input must be dual number or real number!')


if __name__ =="__main__":
    import doctest
    doctest.testmod()