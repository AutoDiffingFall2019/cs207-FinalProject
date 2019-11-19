#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:20:32 2019

@author: shuvomsadhuka
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
    >>> DualNumber(3)
    DualNumber(3,1)
    """

    def __init__(self, real, dual=1):
        assert (isinstance(real, int) or isinstance(real, float) or isinstance(real, DualNumber)), "Check the type of real!"
        #assert (isinstance(dual, float) or isinstance(dual, int)), "Check the type of dual!"
        self._val = real
        self._der = dual

    def val(self):
        return self._val

    def der(self):
        return self._der

# Overloading arithmetic operators

    def __add__(self, other):
        try:
            val2 = self._val + other._val
            der2 = self._der + other._der
            return DualNumber(val2, der2)
        except AttributeError:
            assert(isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            val2 = self._val + other
            der2 = self._der
            return DualNumber(val2, der2)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        try:
            val2 = self._val * other._val
            der2 = self._der * other._val + self._val * other._der
            return DualNumber(val2, der2)
        except AttributeError:
            assert(isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            val2 = self._val * other
            der2 = self._der * other
            return DualNumber(val2, der2)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        try:
            val2 = self._val - other._val
            der2 = self._der - other._der
            return DualNumber(val2, der2)
        except AttributeError:
            assert(isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            val2 = self._val - other
            der2 = self._der
            return DualNumber(val2, der2)

    def __rsub__(self, other):
        try:
            val2 = other._val - self._val
            der2 = other._der - self._der
            return DualNumber(val2, der2)
        except AttributeError:
            assert(isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            val2 = other - self._val
            der2 = -self._der
            return DualNumber(val2, der2)

    def __truediv__(self, other):
        try:
            val2 = self._val / other._val
            der2 = (self._der * other._val - self._val*other._der)/(other._val*other._val)
            return DualNumber(val2, der2)
        except AttributeError:
            assert (isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            val2 = self._val / other
            der2 = self._der / other
            return DualNumber(val2, der2)

    def __rtruediv__(self, other):
        try:
            val2 = other._val / self._val
            der2 = (other._der * self._val - other._val*self._der)/(self._val*self._val)
            return DualNumber(val2, der2)
        except AttributeError:
            assert (isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            val2 = other / self._val
            der2 = -other*self._der / (self._val*self._val)
            return DualNumber(val2, der2)

    def __pow__(self, other):
        try:
            val2 = self._val ** other._val
            der2 = val2*(other._val/self._val*self._der+other._der*np.log(self._val))
            return DualNumber(val2, der2)
        except AttributeError:
            assert (isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            val2 = self._val ** other
            der2 = other * (self._val ** (other - 1)) * self._der
            return DualNumber(val2, der2)

    def __rpow__(self, other):
        try:
            val2 = other._val ** self._val
            der2 = val2*(self._val/other._val*other._der+self._der*np.log(other._val))
            return DualNumber(val2, der2)
        except AttributeError:
            assert (isinstance(other, float) or isinstance(other, int)), "Check the type of objects in function!"
            val2 = other ** self._val
            der2 = other ** self._val * np.log(other)
            return DualNumber(val2, der2)

# Overloading unary operators
    def __pos__(self):
        val2 = self._val
        der2 = self._der
        return DualNumber(val2, der2)

    def __neg__(self):
        val2 = -self._val
        der2 = -self._der
        return DualNumber(val2, der2)

    def __abs__(self):
        val2 = abs(self._val)
        der2 = abs(self._der)
        return DualNumber(val2, der2)

    def __round__(self, n=None):
        val2 = round(self._val, n)
        der2 = round(self._der, n)
        return DualNumber(val2, der2)
#
# if __name__ =="__main__":
#     x=DualNumber(-2.578,-1.2345)
#     y=DualNumber(3,1)
#     f=round(x,2)
#     print(f.val(),f.der())