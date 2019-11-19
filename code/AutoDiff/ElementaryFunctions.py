from DualNumber import DualNumber
import numpy as np

class Sin(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self._val = np.sin(x.val)
            self._der = np.cos(x.val)*x.der
        else:
            self._val = np.sin(x)
            self._der = 0
 
    
class Tan(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self._val = np.tan(x.val)
            self._der = (1+np.tan(x.val)*np.tan(x.val))*x.der
        else:
            self._val = np.tan(x)
            self._der = 0
        

class Cos(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self._val = np.cos(x.val)
            self._der = -1*np.sin(x.val)*x.der
        else:
            self._val = np.cos(x)
            self._der = 0


class Exp(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self._val = np.exp(x.val)
            self._der = np.exp(x.val)*x.der
        else:
            self._val = np.exp(x)
            self._der = 0


class Power(DualNumber):
    def __init__(self, x, n):
        data_type_check(x)
        if data_type_check(x) == 0:
            self._val = x.val**n
            self._der = n*(x.val**(n-1))*x.der
        else:
            self._val = x**n
            self._der = 0
        

class Log(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self._val = np.log(x.val)
            self._der = (1/x.val)*x.der
        else:
            self._val = np.log(x)
            self._der = 0
        
        
class ArcSin(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self._val = np.arcsin(x.val)
            self._der = 1/np.sqrt(1-x.val**2) * x.der
        else:
            self._val = np.arcsin(x)
            self._der = 0
        


class ArcCos(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self._val = np.arccos(x.val)
            self._der = -1/np.sqrt(1-x.val**2) * x.der
        else:
            self._val = np.arccos(x)
            self._der = 0
            
            
class ArcTan(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self._val = np.arctan(x.val)
            self._der = 1/np.sqrt(1+x.val**2) * x.der
        else:
            self._val = np.arctan(x)
            self._der = 0


class Sqrt(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self._val = np.sqrt(x.val)
            self._der = 1/(2*np.sqrt(x.val)) * x.der
        else:
            self._val = np.sqrt(x)
            self._der = 0
        
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