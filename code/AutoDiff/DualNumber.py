#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:20:32 2019

@author: shuvomsadhuka
"""

import numpy as np
import doctest

class DualNumber():
    '''
    Description: a class to hold dual number representations of vectors/scalars.
    '''

    def __init__(self, real, dual=1):
        assert (isinstance(real, int) or isinstance(real, float) or isinstance(real, DualNumber)), "Check the type of real!"
        #assert (isinstance(dual, float) or isinstance(dual, int)), "Check the type of dual!"
        self._val = real
        self._der = dual

    @property
    def val(self):
        return self._val
    
    @property
    def der(self):
        return self._der
    
    def __repr__(self):
        print('Derivative: {0:5f}\n'.format(self.der) + 'Value: {0:.2f}'.format(self.val))
        
    def __str__(self):
        return('Derivative: {0:.2f}\n'.format(self.der) + 'Value: {0:.2f}'.format(self.val))

# Overloading arithmetic operators

    def __add__(self, other):
        try:
            val2 = self.val + other.val
            der2 = self.der + other.der
            return DualNumber(val2, der2)
        except AttributeError:
            assert(isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            val2 = self.val + other
            der2 = self.der
            return DualNumber(val2, der2)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        try:
            val2 = self.val * other.val
            der2 = self.der * other.val + self.val * other.der
            return DualNumber(val2, der2)
        except AttributeError:
            assert(isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            val2 = self.val * other
            der2 = self.der * other
            return DualNumber(val2, der2)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        try:
            val2 = self.val - other.val
            der2 = self.der - other.der
            return DualNumber(val2, der2)
        except AttributeError:
            assert(isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            val2 = self.val - other
            der2 = self.der
            return DualNumber(val2, der2)

    def __rsub__(self, other):
        try:
            val2 = other.val - self.val
            der2 = other.der - self.der
            return DualNumber(val2, der2)
        except AttributeError:
            assert(isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            val2 = other - self.val
            der2 = -self.der
            return DualNumber(val2, der2)

    def __truediv__(self, other):
        try:
            val2 = self.val / other.val
            der2 = (-self.val * other.der + self.der*other.val)/(other.val*other.val)
            return DualNumber(val2, der2)
        except AttributeError:
            assert (isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            val2 = self.val / other
            der2 = self.der / other
            return DualNumber(val2, der2)

    def __rtruediv__(self, other):
        try:
            val2 = other.val / self.val
            der2 = (other.der * self.val - other.val*self.der)/(self.val*self.val)
            return DualNumber(val2, der2)
        except AttributeError:
            assert (isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            val2 = other / self.val
            der2 = -other*self.der / (self.val*self.val)
            return DualNumber(val2, der2)

    def __pow__(self, other):
        try:
            val2 = self.val ** other.val
            der2 = val2*(other.val/self.val*self.der+other.der*np.log(self.val))
            return DualNumber(val2, der2)
        except AttributeError:
            assert (isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            val2 = self.val ** other
            der2 = other * (self.val ** (other - 1)) * self.der
            return DualNumber(val2, der2)


    def __rpow__(self, other):
        try:
            val2 = other.val ** self.val
            der2 = val2*(self.val/other.val*other.der+self.der*np.log(other.val))
            return DualNumber(val2, der2)
        except AttributeError:
            assert (isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            val2 = other ** self.val
            der2 = other ** self.val * np.log(other)
            return DualNumber(val2, der2)

# Overloading unary operators
    def __pos__(self):
        val2 = self.val
        der2 = self.der
        return DualNumber(val2, der2)

    def __neg__(self):
        val2 = -self.val
        der2 = -self.der
        return DualNumber(val2, der2)

    def __abs__(self):
        val2 = abs(self.val)
        if self.val >= 0:
            der2 = abs(self.der)
        else:
            der2 = self.der
        return DualNumber(val2, der2)

    def __round__(self, n=None):
        val2 = round(self.val, n)
        der2 = round(self.der, n)
        return DualNumber(val2, der2)

if __name__ =="__main__":
    x=DualNumber(-2.578,-1.2345)
    y=DualNumber(3,1)
    f=round(x,2)
    print(f.val,f.der)