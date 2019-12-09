#!/anaconda3/bin/python
# -*- coding: utf-8 -*-

from DualNumber_with_Reverse import DualNumber
import numpy as np

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

def Log(x):
    '''
    >>> print(Log(DualNumber(5,1)))
    Derivative: 0.20
    Value: 1.61
    '''
    if data_type_check(x) == 0:
        if x._rev:
            z=DualNumber(np.log(x._val),Reverse=True)
            x.children.append(((1/x._val),z))
            return z          
        return DualNumber(np.log(x._val),(1/x._val)*x._der)
    else:
        return DualNumber(np.log(x),0)

        
def ArcSin(x):
    '''
    >>> print(ArcSin(DualNumber(0.5)))
    Derivative: 1.15
    Value: 0.52
    '''
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
    '''
    if data_type_check(x) == 0:
        if x._rev:
            z=DualNumber(np.sqrt(x._val),Reverse=True)
            x.children.append((1/(2*np.sqrt(x._val)),z))
            return z             
        return DualNumber(np.sqrt(x._val),1/(2*np.sqrt(x._val)) * x._der)
    else:
        return DualNumber(np.sqrt(x),0)
        
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
    x = DualNumber(0.5,Reverse=True)
    y = DualNumber(4.2,Reverse=True)
    z = x * y + Sin(x)
    z.set_der(value=1.0)
    print('value of x*y + sin(x) evaluated at x=0.5, y=4.2: {}\nOur implementation gives: {}'.format(0.5 * 4.2 + np.sin(0.5), z.val))
    print('value of dz/dx = y + cos(x) evaluated at x=0.5, y=4.2: {}\nReverse method of our implementation: {}'.format(4.2 + np.cos(0.5), x.der))
    print('value of dz/dy = x evaluated at x=0.5, y=4.2: {}\nReverse method of our implementation: {}'.format(0.5, y.der))

    #import doctest
    #doctest.testmod()