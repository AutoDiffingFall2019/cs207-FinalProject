#!/anaconda3/bin/python
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo code for using AD package
Created on Thu Nov 14 13:56:17 2019
"""

import AD.ElementaryFunctions as EF
from AD.DualNumber import DualNumber
from AD.Parallelized import Parallelized_AD
func = ['_x + sin(_y)*_z', '_x + sin(_y)*exp(_z)']
PAD = Parallelized_AD(fun = func, var = ['x', 'y', 'z'])

print("JACOBIAN: ")
print(PAD.get_Jacobian([1,2,3]))

print("VALUE: ")
print(PAD.get_value([1,2,3]))

PAD.add_var('w')
PAD.add_function('_w')

print('JACOBIAN FORWARD:')
print(PAD.get_Jacobian([1,2,3,4], forward=True))

print('JACOBIAN REVERSE:')
print(PAD.get_Jacobian([1,2,3,4], forward=False))

print("VALUE: ")
print(PAD.get_value([1,2,3,4]))