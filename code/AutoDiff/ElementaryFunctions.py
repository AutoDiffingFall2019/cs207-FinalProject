from DualNumber import DualNumber
import numpy as np

class Sin(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        self.val = np.sin(x.val)
        self.der = np.cos(x.val)*x.der
 
    
class Tan(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        self.val = np.tan(x.val)
        self.der = (1+np.tan(x.val)*np.tan(x.val))*x.der
        

class Cos(DualNumber):
    def __init__(self, x):
        print(x.val)
        print(x.der)
        data_type_check(x)
        self.val = np.cos(x.val)
        self.der = -1*np.sin(x.val)*x.der


class Exp(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        self.val = np.exp(x.val)
        self.der = np.exp(x.val)*x.der


class Power(DualNumber):
    def __init__(self, x, n):
        data_type_check(x)
        self.val = x.val**n
        self.der = n*(x.val**(n-1))*x.der
        

class Log(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        self.val = np.log(x.val)
        self.der = (1/x.val)*x.der
        
        
class ArcSin(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        self.val = np.arcsin(x.val)
        self.der = 1/np.sqrt(1-x.val**2) * x.der


class ArcCos(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        self.val = np.arccos(x.val)
        self.der = -1/np.sqrt(1-x.val**2) * x.der
            
            
class ArcTan(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        self.val = np.arctan(x.val)
        self.der = 1/np.sqrt(1+x.val**2) * x.der


class Sqrt(DualNumber):
    def __init__(self, x):
        data_type_check(x)
        self.val = np.sqrt(x.val)
        self.der = 1/(2*np.sqrt(x.val)) * x.der
        
        
def data_type_check(x):
   try:
       float(x.val)+float(x.der)
   except ValueError:
       print("Input has to have attributes x.val and x.der")
    