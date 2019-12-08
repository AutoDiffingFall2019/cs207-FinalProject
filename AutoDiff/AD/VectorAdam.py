from DualNumber_with_reverse import DualNumber
import ElementaryFunctions_with_reverse as EF
from Parallelized import Parallelized_AD
import numpy as np

class Adam():
    def __init__(self, func, var_names, init_val, tol=1e-8, max_iter=1e5):
        assert (len(var_names) == len(init_val)), "Not a valid initial value!"
        self.init_val = init_val
        self.func = func
        self.var_names = var_names
        self.curr_val = init_val
        self._tol = tol
        self._max_iter = max_iter
        
        
    def run(self):
        PAD=Parallelized_AD(fun=self.func,var=self.var_names)
        old_val = self.curr_val
        
        self.curr_val += PAD.get_Jacobian(self.curr_val)
        iterate = 0
        
        while np.linalg.norm(old_val - self.curr_val) > self._tol and iterate < self._max_iter:
            print(self.curr_val.shape)
            old_val = self.curr_val
            self.curr_val += PAD.get_Jacobian(self.curr_val)
    
            
    

if __name__=='__main__':
    func=['sin(_x**2+_y)*_z']
    var_names = ['x', 'y', 'z']
    AD = Adam(func, var_names, [1,1,1])
    AD.run()
    optimized_location = AD.curr_val
    print(optimized_location)