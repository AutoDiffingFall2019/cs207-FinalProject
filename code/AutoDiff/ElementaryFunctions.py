from DualNumber import DualNumber
import numpy as np

class Sin(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        self._val = np.sin(x._val)
        self._der = np.cos(x._val)*x._der
 
    
class Tan(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        self._val = np.tan(x._val)
        self._der = (1+np.tan(x._val)*np.tan(x._val))*x._der
        

class Cos(DualNumber):
    def __init__(self, x):
        print(x._val)
        print(x._der)
        data_type_check(x)
        self._val = np.cos(x._val)
        self._der = -1*np.sin(x._val)*x._der


class Exp(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        self._val = np.exp(x._val)
        self._der = np.exp(x._val)*x._der


class Power(DualNumber):
    def __init__(self, x, n):
        data_type_check(x)
        self._val = x._val**n
        self._der = n*(x._val**(n-1))*x._der
        

class Log(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        self._val = np.log(x._val)
        self._der = (1/x._val)*x._der
        
        
class ArcSin(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        self._val = np.arcsin(x._val)
        self._der = 1/np.sqrt(1-x._val**2) * x._der


class ArcCos(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        self._val = np.arccos(x._val)
        self._der = -1/np.sqrt(1-x._val**2) * x._der
            
            
class ArcTan(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        self._val = np.arctan(x._val)
        self._der = 1/np.sqrt(1+x._val**2) * x._der


class Sqrt(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        self._val = np.sqrt(x._val)
        self._der = 1/(2*np.sqrt(x._val)) * x._der
        
        
def data_type_check(x):
   try:
       float(x._val)+float(x._der)
   except ValueError:
       print("Input has to have attributes x.val and x.der")
    