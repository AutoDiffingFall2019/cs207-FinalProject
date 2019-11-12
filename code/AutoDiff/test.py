# -*- coding: utf-8 -*-

from DualNumber import DualNumber
from ElementaryFunctions import *

a = DualNumber(5)

b = Exp(a)
c = Cos(Sqrt(b)) + b**2
d = Log(c)

print(c.val, c.der)
print(d.val, d.der)