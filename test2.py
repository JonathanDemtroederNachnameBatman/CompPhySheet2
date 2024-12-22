import numpy as np

import network
from numba import jit

@jit() # doesn't work without jit
def set_seed():
    np.random.seed(123456789)

set_seed()

p = network.Protein()
network.plot_random_walk(p.chain)
print(np.flip(p.grid.T, axis=0))
x = -1
y = 1
shift = p.check_expand(x, y)
print(shift)
x -= shift[0]
y -= shift[1]
print(np.flip(p.grid.T, axis=0))
#network.plot_random_walk(p.chain)
print(p.grid[11][1])
