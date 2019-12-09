#!/anaconda3/bin/python
# -*- coding: utf-8 -*-

from DualNumber import DualNumber
import ElementaryFunctions as EF
from Parallelized import Parallelized_AD
import numpy as np

class Adam():
    """
    DESCRIPTION
    =======
    A class to run basic gradient descent LOCAL optimization using our autoamtic 
    differentation package.  
    
    INPUTS
    =======
    func: array
        Stores the function which we wish to optimize.
    var_names: array
        Stores list of variables (will differentiate function wrt each)
    init_val: array
        Stores initial guess for minimum/maxmium of the function
    RETURNS
    ========
    curr_val: minumum/maximum found after optimization procedure
    

    EXAMPLES
    =========
    >>> func=['sin(_x**2+_y)*_z']
    >>> var_names = ['x', 'y', 'z']
    >>> optimize = Adam(func, var_names, [1,1,1])
    >>> optimize.run()
    >>> optimize.curr_val
    array([[1.81201138, 1.41061216, 1.08883911]])
    >>> optimize.run(minimum=False)
    >>> optimize.curr_val
    array([[1.73944269, 1.39058782, 0.08900824]])
    """
    def __init__(self, func, var_names, init_val, tol=1e-10, max_iter=1e5):
        assert (len(var_names) == len(init_val)), "Not a valid initial value!"
        self.init_val = init_val
        self.func = func
        self.var_names = var_names
        self.curr_val = init_val
        self._tol = tol
        self._max_iter = max_iter
        
        
    def run(self, minimum=True):
        # note run() takes an argument minimum, if true, finds minimum
        # else finds maximum.  Default: minumum
        PAD=Parallelized_AD(fun=self.func,var=self.var_names)
        old_val = self.curr_val
        if minimum:
            self.curr_val -= PAD.get_Jacobian(self.curr_val)
        else:
            self.curr_val += PAD.get_Jacobian(self.curr_val)
        iterate = 0
        
        while np.linalg.norm(old_val - self.curr_val) > self._tol and iterate < self._max_iter:
            old_val = self.curr_val
            if minimum:
                self.curr_val -= PAD.get_Jacobian(self.curr_val)
            else:
                self.curr_val += PAD.get_Jacobian(self.curr_val)
            iterate += 1
    

if __name__=='__main__':
    import doctest
    doctest.testmod()