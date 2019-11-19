from DualNumber import DualNumber
import numpy as np

def Sin(x):
    if data_type_check(x) == 0:
        return DualNumber(np.sin(x._val),np.cos(x._val)*x._der)
    else:
        return DualNumber(np.sin(x),0)

    
def Tan(x):
    if data_type_check(x) == 0:
        return DualNumber(np.tan(x._val),(1+np.tan(x._val)*np.tan(x._val))*x._der)
    else:
        return DualNumber(np.tan(x._val),0)


def Cos(x):
    if data_type_check(x) == 0:
        return DualNumber(np.cos(x._val),-1*np.sin(x._val)*x._der)
    else:
        return DualNumber(np.cos(x),0)

def Exp(x):
    if data_type_check(x) == 0:
        return DualNumber(np.exp(x._val),np.exp(x._val)*x._der)
    else:
        return DualNumber(np.exp(x),0)

def Power(x,n):
    if data_type_check(x) == 0:
        return DualNumber(x._val**n,n*(x._val**(n-1))*x._der)
    else:
        return DualNumber(x**n,0)

def Log(x):
    if data_type_check(x) == 0:
        return DualNumber(np.log(x._val),(1/x._val)*x._der)
    else:
        return DualNumber(np.log(x),0)

        
def ArcSin(x):
    if data_type_check(x) == 0:
        return DualNumber(np.arcsin(x._val),1/np.sqrt(1-x._val**2) * x._der)
    else:
        return DualNumber(np.arcsin(x),0)


def ArcCos(x):
    if data_type_check(x) == 0:
        return DualNumber(np.arccos(x._val),-1/np.sqrt(1-x._val**2) * x._der)
    else:
        return DualNumber(np.arccos(x),0)
            
def ArcTan(x):
    if data_type_check(x) == 0:
        return DualNumber(np.arctan(x._val),1/np.sqrt(1+x._val**2) * x._der)
    else:
        return DualNumber(np.arctan(x),0)

def Sqrt(x):
    if data_type_check(x) == 0:
        return DualNumber(np.sqrt(x._val),1/(2*np.sqrt(x._val)) * x._der)
    else:
        return DualNumber(np.sqrt(x),0)
        
def data_type_check(x):
   try:
       float(x._val)+float(x._der)
       return 0  # returns 0 if x is DualNumber
   except AttributeError:
       try:
           float(x)
           return 1 # returns 1 if x is real
       except:
           raise AttributeError('Input must be dual number or real number!')    
