from DualNumber import DualNumber
import numpy as np

class Sin(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self.val = np.sin(x.val)
            self.der = np.cos(x.val)*x.der
        else:
            self.val = np.sin(x)
            self.der = 0
 
    
class Tan(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self.val = np.tan(x.val)
            self.der = (1+np.tan(x.val)*np.tan(x.val))*x.der
        else:
            self.val = np.tan(x)
            self.der = 0
        

class Cos(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self.val = np.cos(x.val)
            self.der = -1*np.sin(x.val)*x.der
        else:
            self.val = np.cos(x)
            self.der = 0


class Exp(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self.val = np.exp(x.val)
            self.der = np.exp(x.val)*x.der
        else:
            self.val = np.exp(x)
            self.der = 0


class Power(DualNumber):
    def __init__(self, x, n):
        data_type_check(x)
        if data_type_check(x) == 0:
            self.val = x.val**n
            self.der = n*(x.val**(n-1))*x.der
        else:
            self.val = x**n
            self.der = 0
        

class Log(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self.val = np.log(x.val)
            self.der = (1/x.val)*x.der
        else:
            self.val = np.log(x)
            self.der = 0
        
        
class ArcSin(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self.val = np.arcsin(x.val)
            self.der = 1/np.sqrt(1-x.val**2) * x.der
        else:
            self.val = np.arcsin(x)
            self.der = 0
        


class ArcCos(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self.val = np.arccos(x.val)
            self.der = -1/np.sqrt(1-x.val**2) * x.der
        else:
            self.val = np.arccos(x)
            self.der = 0
            
            
class ArcTan(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self.val = np.arctan(x.val)
            self.der = 1/np.sqrt(1+x.val**2) * x.der
        else:
            self.val = np.arctan(x)
            self.der = 0


class Sqrt(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        if data_type_check(x) == 0:
            self.val = np.sqrt(x.val)
            self.der = 1/(2*np.sqrt(x.val)) * x.der
        else:
            self.val = np.sqrt(x)
            self.der = 0
        
def data_type_check(x):
   try:
       float(x.val)+float(x.der)
       return 0  # returns 0 if x is DualNumber
   except AttributeError:
       try:
           float(x)
           return 1 # returns 1 if x is real
       except:
           raise AttributeError('Input must be dual number or real number!')