import numpy as np
from numba import jit, double
from numba.experimental import jitclass

@jitclass([('matrix', double[:,:])])
class Interaction:
    def __init__(self, size=20, sigma=1/np.sqrt(2), mean=-3, const=False, const_value = -3):

        if const:
            self.matrix = np.full(shape=(size,size), fill_value=const_value, dtype=double)
        else:
            self.matrix = np.random.normal(mean, sigma, size=(size, size))

    def eigenvalues(self):
        return np.linalg.eigvals(self.matrix+ 0j) # add imaginary part to compensate for possible (forbidden) domain changes
