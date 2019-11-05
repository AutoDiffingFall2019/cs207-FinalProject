from DualNumber import DualNumber
import numpy as np

class Sin(DualNumber):
    def __init___(self, x):
        self.value = np.sin(x.value)
        self.der = np.cos(x.value)*x.der
 
    
class Tan(DualNumber):
    def __init___(self, x):
        self.value = np.tan(x.value)
        self.der = (1+np.tan(x.value)*np.tan(x.value))*x.der
        

class Cos(DualNumber):
    def __init___(self, x):
        self.value = np.cos(x.value)
        self.der = -1*np.sin(x.value)*x.der


class Exp(DualNumber):
    def __init___(self, x):
        self.value = np.exp(x.value)
        self.der = np.exp(x.value)*x.der


class Power(DualNumber):
    def __init___(self, x, n):
        self.value = x.value**n
        self.der = n*(x.value**(n-1))*x.der
        

class Log(DualNumber):
    def __init___(self, x):
        self.value = np.log(x.value)
        self.der = (1/x.value)*x.der
        
        
class ArcSin(DualNumber):
    def __init___(self, x):
        self.value = np.arcsin(x.value)
        try:
            self.der = 1/np.sqrt(1-x.value**2) * x.der
        except Exception as e:
            print(f'ArcSin has domain (-1,1)!{e}')


class ArcCos(DualNumber):
    def __init___(self, x):
        self.value = np.arccos(x.value)
        assert abs(x.value) <= 1
        self.der = -1/np.sqrt(1-x.value**2) * x.der

            
            
class ArcTan(DualNumber):
    def __init___(self, x):
        self.value = np.arctan(x.value)
        self.der = 1/np.sqrt(1+x.value**2) * x.der


class Sqrt(DualNumber):
    def __init__(self, x):
        self.value = np.sqrt(x.value)
        self.der = 1/(2*np.sqrt(x.value)) * x.der
        
def data_tyoe_check():
    # TODO
    pass
    