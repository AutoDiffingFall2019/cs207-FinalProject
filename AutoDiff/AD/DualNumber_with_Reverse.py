#!/anaconda3/bin/python
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:20:32 2019

"""

import numpy as np


class DualNumber():

    """Returns DualNumber and defines the overloading of arithmetric and unary operators.
    INPUTS
    =======
    real: float
      This stores the value of the function for which the derivative would be evaluated at.
    dual: float, optional, default value is 1
      This stores the value for the derivative of the function.
    Reverse: Boolean, optional, if set to true the gradient will be calculated based on reverse mode
        Default value is false, forward mode is used
    RETURNS
    ========
    DualNumber: An instance with methods such as val() and der() to get out the current value and derivative of function respectively.
    NOTES
    =====
    PRE:
        - real, dual have numeric types
        - input of real is needed but input of real is optional
    POST:
        - real will be stored in the private attribute self._val and holds the current value of the function
        - dual will be stored in the private attribute self._der and holds the current derivative value of the function
        - raises a AttributeError exception if either real or dual is not of numeric type
        - returns a DualNumber instance
        - defines the overloading of arithmetic operators and unary operators
    EXAMPLES
    =========
    >>> [DualNumber(3,2).val, DualNumber(3,2).der]
    [3, 2]
    >>> [DualNumber(2.3,2).val, DualNumber(2.3,2).der]
    [2.3, 2]
    >>> [DualNumber(2,2.3).val, DualNumber(2,2.3).der]
    [2, 2.3]
    >>> [DualNumber(2.3,2.3).val, DualNumber(2.3,2.3).der]
    [2.3, 2.3]
    >>> print(DualNumber(2.3,2.3))
    Derivative: 2.30
    Value: 2.30
    """

    def __init__(self, real, dual=1,Reverse=False):
        #print(real)
        assert (isinstance(real, int) or isinstance(real, float)), "Check the type of real!"
        assert (isinstance(dual, float) or isinstance(dual, int)), "Check the type of dual!"
        self._val = real
        self._der = dual
        self._rev=Reverse
        if Reverse is True:
            self._der = None
            self.children = []
            self._rev=True
    @property
    def val(self):
        return self._val
    
    @property
    def der(self):
        if self._der is None:
            self._der = sum(weight * var.der for weight, var in self.children)
        return self._der    
    @property
    def jacobian(self):
        return self.der
    
    def __repr__(self):
        return 'Derivative: {0:.2f}'.format(self.der) + '\n' + 'Value: {0:.2f}'.format(self.val)
        
    def __str__(self):
        return('Derivative: {0:.2f}'.format(self.der) + '\n' + 'Value: {0:.2f}'.format(self.val))
    def set_der(self,value=None):
        #Set derivative of current dual number
        if value: self._der=value        
    def set_val(self,value=None):
        #Set val of current dual number
        if value: self._val=value        
# Overloading arithmetic operators

    def __add__(self, other):
        '''
        >>> print(DualNumber(5,1)+3)
        Derivative: 1.00
        Value: 8.00
        '''
        try:
            if self._rev:
                z = DualNumber(self._val + other._val,Reverse=True)
                self.children.append((1.0, z))
                other.children.append((1.0, z))     
                return z
            else:
                val2 = self.val + other.val
                der2 = self.der + other.der
                return DualNumber(val2, der2)
        except AttributeError:
            assert(isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            if self._rev:
                z = DualNumber(self._val + other,Reverse=True)
                self.children.append((1.0, z))
                return z                    
            else:
                return DualNumber(self.val + other, self.der)

    def __radd__(self, other):
        '''
        >>> print(3+DualNumber(5,1))
        Derivative: 1.00
        Value: 8.00
        '''
        return self.__add__(other)

    def __mul__(self, other):
        '''
        >>> print(DualNumber(5,1)*3)
        Derivative: 3.00
        Value: 15.00
        '''
        try:
            if self._rev:
                z = DualNumber(self._val * other._val,Reverse=True)
                self.children.append((other.val, z))
                other.children.append((self.val, z)) 
                return z
            else:
                val2 = self.val * other.val
                der2 = self.der * other.val + self.val * other.der
                return DualNumber(val2, der2)
        except AttributeError:
            assert(isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            if self._rev:
                z = DualNumber(self._val * other,Reverse=True)
                self.children.append((other, z))
                return z
            val2 = self.val * other
            der2 = self.der * other
            return DualNumber(val2, der2)

    def __rmul__(self, other):
        '''
        >>> print(3*DualNumber(5,1))
        Derivative: 3.00
        Value: 15.00
        '''
        return self.__mul__(other)

    def __sub__(self, other):
        '''
        >>> print(DualNumber(5,1)-3)
        Derivative: 1.00
        Value: 2.00
        '''
        try:
            if self._rev:
                z = DualNumber(self.val - other.val,Reverse=True)
                self.children.append((1, z))
                other.children.append((-1, z))
                return z  
            else:
                val2 = self.val - other.val
                der2 = self.der - other.der
                return DualNumber(val2, der2)
        except AttributeError:
            assert(isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            if self._rev:
                z = DualNumber(self._val - other,Reverse=True)
                self.children.append((1, z))
                return z 
            else:
                val2 = self.val - other
                der2 = self.der
                return DualNumber(val2, der2)

    def __rsub__(self, other):
        '''
        >>> print(3-DualNumber(5,1))
        Derivative: -1.00
        Value: -2.00
        '''
        try:
            if self._rev:
                z = DualNumber(other.val - self.val,Reverse=True)
                self.children.append((-1, z))
                other.children.append((1, z))
                return z                
            else:
                val2 = other.val - self.val
                der2 = other.der - self.der
                return DualNumber(val2, der2)
        except AttributeError:
            assert(isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            if self._rev:
                z = DualNumber(other - self.val,Reverse=True)
                self.children.append((-1, z))
                return z
            else:
                val2 = other - self.val
                der2 = -self.der
                return DualNumber(val2, der2)

    def __truediv__(self, other):
        '''
        >>> print(DualNumber(5,1)/2)
        Derivative: 0.50
        Value: 2.50
        '''
        try:
            if self._rev:
                z = DualNumber(self.val / other.val,Reverse=True)
                self.children.append((1/other.val , z))
                other.children.append((-self.val/(other.val*other.val) , z))
                return z 
            else:                 
                val2 = self.val / other.val
                der2 = (-self.val * other.der + self.der*other.val)/(other.val*other.val)
                return DualNumber(val2, der2)
        except AttributeError:
            assert (isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            if self._rev:
                z = DualNumber(self.val / other,Reverse=True)
                self.children.append((1/other , z))
                return z
            else:
                val2 = self.val / other
                der2 = self.der / other
                return DualNumber(val2, der2)

    def __rtruediv__(self, other):
        '''
        >>> print(2/DualNumber(5,1))
        Derivative: -0.08
        Value: 0.40
        '''
        try:
            if self._rev:
                z = DualNumber(other.val/self.val,Reverse=True)
                self.children.append((-other.val/self.val/self.val , z))
                other.children.append((1/self.val , z))
                return z            
            val2 = other.val / self.val
            der2 = (other.der * self.val - other.val*self.der)/(self.val*self.val)
            return DualNumber(val2, der2)
        except AttributeError:
            assert (isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            if self._rev:
                z = DualNumber(other/self.val,Reverse=True)
                self.children.append((-other/self.val/self.val , z))
                return z             
            val2 = other / self.val
            der2 = -other*self.der / (self.val*self.val)
            return DualNumber(val2, der2)

    def __pow__(self, other):
        '''
        >>> print(DualNumber(5,1)**2)
        Derivative: 10.00
        Value: 25.00
        '''
        try:
            if self._rev:
                z = DualNumber(self.val ** other.val,Reverse=True)
                self.children.append(((self.val ** other.val)*other.val/self.val , z))
                other.children.append(((self.val ** other.val)*np.log(self.val), z))
                return z      
            else:
                val2 = self.val ** other.val
                der2 = val2*(other.val/self.val*self.der+other.der*np.log(self.val))
                return DualNumber(val2, der2)
        except AttributeError:
            assert (isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            if self._rev:
                z = DualNumber(self.val ** other,Reverse=True)
                self.children.append(((self.val ** other)*other/self.val , z))
                return z         
            val2 = self.val ** other
            der2 = other * (self.val ** (other - 1)) * self.der
            return DualNumber(val2, der2)


    def __rpow__(self, other):
        '''
        >>> print(2**DualNumber(5,1))
        Derivative: 22.18
        Value: 32.00
        '''
        try:
            if self._rev:
                z = DualNumber(other.val ** self.val,Reverse=True)
                other.children.append(((other.val ** self.val)*self.val/other.val , z))
                self.children.append(((other.val ** self.val)*np.log(other.val), z))
                return z   
            else:                
                val2 = other.val ** self.val
                der2 = val2*(self.val/other.val*other.der+self.der*np.log(other.val))
                return DualNumber(val2, der2)
        except AttributeError:
            assert (isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            if self._rev:
                z = DualNumber(other ** self.val,Reverse=True)
                self.children.append((other ** self.val * np.log(other), z))
                return z   
            val2 = other ** self.val
            der2 = other ** self.val * np.log(other) * self.der
            return DualNumber(val2, der2)

# Overloading unary operators
    def __pos__(self):
        '''
        >>> print(+DualNumber(-5,-1))
        Derivative: -1.00
        Value: -5.00
        '''
        if self._rev:
            z = DualNumber(self.val,Reverse=True)
            self.children.append((1, z))
            return z
        else:
            val2 = self.val
            der2 = self.der
            return DualNumber(val2, der2)

    def __neg__(self):
        '''
        >>> print(-DualNumber(-5,-1))
        Derivative: 1.00
        Value: 5.00
        '''
        if self._rev:
            z = DualNumber(-self.val,Reverse=True)
            self.children.append((-1, z))
            return z    
        else:
            val2 = -self.val
            der2 = -self.der
            return DualNumber(val2, der2)

    def __abs__(self):
        '''
        >>> print(abs(DualNumber(-5,-1)))
        Derivative: 1.00
        Value: 5.00
        '''
        if self._rev:
            z = DualNumber(abs(self.val),Reverse=True)
            if self.val >= 0:
                self.children.append(( 1, z))
            else:
                self.children.append((-1, z))
            return z                
        else:
            val2 = abs(self.val)
            if self.val >= 0:
                der2 = self.der
            else:
                der2 = -1 * self.der
            return DualNumber(val2, der2)

    def __round__(self, n=None):
        '''
        >>> print(round(DualNumber(-5.234,-1.256),2))
        Derivative: -1.26
        Value: -5.23
        '''
        if self._rev:
            z = DualNumber(round(self.val, n),Reverse=True)
            self.children.append((1, z))
            return z    
        else:
            val2 = round(self.val, n)
            der2 = round(self.der, n)
            return DualNumber(val2, der2)
if __name__ =="__main__":
    # x=DualNumber(-2.578,-1.2345)
    # y=DualNumber(3,1)
    # print(x)
    # f=round(x,2)
    # print(f.val,f.der)
    import doctest
    doctest.testmod()
    
