# -*- coding: utf-8 -*-

from DualNumber import DualNumber
from ElementaryFunctions import *

a = DualNumber(5)

b = Cos(a)
c = Sqrt(Sin(b)) + b
d = c**5

print(c.val, c.der)
print(d.val, d.der)