from DualNumber import DualNumber
import numpy as np

class Sin(DualNumber):
    def __init__(self, x):
	data_type_check(x)
	if data_type_check(x) == 0:
            self._val = np.sin(x._val)
            self._der = np.cos(x._val)*x._der
        else:
            self._val = np.sin(x)
            self._der = 0
    
class Tan(DualNumber):
    def __init__(self, x):
        data_type_check(x)
	if data_type_check(x) == 0:
            self._val = np.tan(x._val)
            self._der = (1+np.tan(x._val)*np.tan(x._val))*x._der
        else:
            self._val = np.tan(x)
            self._der = 0	

class Cos(DualNumber):
    def __init__(self,x):
	data_type_check(x)
        if data_type_check(x) == 0:
            self._val = np.cos(x._val)
            self._der = -1*np.sin(x._val)*x._der
        else:
            self._val = np.cos(x)
            self._der = 0

class Exp(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self._val = np.exp(x._val)
            self._der = np.exp(x._val)*x._der
        else:
            self._val = np.exp(x)
            self._der = 0

class Power(DualNumber):
    def __init__(self, x, n):
        data_type_check(x)
        if data_type_check(x) == 0:
            self._val = x._val**n
            self._der = n*(x._val**(n-1))*x._der
        else:
            self._val = x**n
            self._der = 0

class Log(DualNumber):
    def __init__(self, x):
	data_type_check(x)
        if data_type_check(x) == 0:
            self._val = np.log(x._val)
            self._der = (1/x._val)*x._der
        else:
            self._val = np.log(x)
            self._der = 0 
        
class ArcSin(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self._val = np.arcsin(x._val)
            self._der = 1/np.sqrt(1-x._val**2) * x._der
        else:
            self._val = np.arcsin(x)
            self._der = 0

class ArcCos(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self._val = np.arccos(x._val)
            self._der = -1/np.sqrt(1-x._val**2) * x._der
        else:
            self._val = np.arccos(x)
            self._der = 0            
            
class ArcTan(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self._val = np.arctan(x._val)
            self._der = 1/np.sqrt(1+x._val**2) * x._der
        else:
            self._val = np.arctan(x)
            self._der = 0

class Sqrt(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self._val = np.sqrt(x._val)
            self._der = 1/(2*np.sqrt(x._val)) * x._der
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
