import numba
import numpy as np

import network
from numba import jit, void

@jit() # doesn't work without jit
def set_seed():
    np.random.seed(123456789)

set_seed()
p = network.Protein()
network.plot_protein(p)

@jit(void(network.Protein.class_type.instance_type, numba.int32))
def fold(protein, amount):
    for i in range(amount):
        protein.random_fold_step()

fold(p, 469435)
network.plot_protein(p)
fold(p, 1)
network.plot_protein(p)
fold(p, 1)
network.plot_protein(p)
p.verify_chain_grid()


"""
set_seed()

p = network.Protein()
network.plot_random_walk(p.chain)
print(np.flip(p.grid.T, axis=0))
x = -1
y = 1
shift = p.check_bounds(x, y)
print(shift)
x -= shift[0]
y -= shift[1]
print(np.flip(p.grid.T, axis=0))
#network.plot_random_walk(p.chain)
print(p.grid[11][1])
"""


