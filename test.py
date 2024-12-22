
import network
import numpy as np
from numba import jit

@jit() # doesn't work without jit
def set_seed():
    np.random.seed(123456789)

set_seed()
p = network.Protein()

network.plot_random_walk(p.chain)
print(p.chain)
print(np.flip(p.grid.T, axis=0))

def fold(x, y):
    print(p.fold_step_at(x, y, False))
    network.plot_random_walk(p.chain)

fold(3, 5) # True
fold(5, 5) # False
fold(5, 3) # True
fold(3, 3) # True

print(p.chain)
print(np.flip(p.grid.T, axis=0)) # print correct representation of grid
