#!/anaconda3/bin/python
# -*- coding: utf-8 -*-

from AD.DualNumber import DualNumber
import AD.ElementaryFunctions as EF
from AD.Parallelized import Parallelized_AD
from AD.Adam import Adam_optimizer
import numpy as np
import pytest
def test_Adam():
    def Demo_gradient(x):
        x=DualNumber(x);
        y=-EF.Sin(x)*EF.Cos(x)*EF.Tan(x)+EF.Exp(x)*EF.Log(x)*EF.Sqrt(x)*2
        return y.der #y=sin(x)cos(x)tan(x)-2exp(x)log(x)sqrt(x)
    Ad=Adam_optimizer(Demo_gradient)
    Ad.set_objective(Demo_gradient)
    optimized_location=Ad.optimize()
    assert(np.abs(0.37667568453589606-optimized_location)<1e-5)

test_Adam()
