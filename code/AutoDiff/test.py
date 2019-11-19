# -*- coding: utf-8 -*-

from DualNumber import DualNumber
from ElementaryFunction2 import *

a = DualNumber(5)

b = Exp(a)
c = Cos(Sqrt(b)) + b**2
d = Log(Sqrt(25))*Power(5, 4)/Exp(a)+Log(Log(c))

print(c)