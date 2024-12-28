import network
import numpy as np
from numba import jit

@jit() # doesn't work without jit
def set_seed():
    np.random.seed(1)

#set_seed()
p = network.create_protein(interaction_type='normal')

network.plot_random_walk(p.chain)
print(p.chain)
#print(np.flip(p.grid.T, axis=0))

def fold(x, y):
    print(p.fold_step_at(x, y, False))
    network.plot_random_walk(p.chain)

print('START LOOP')
for i in range(10):
    p.random_fold_step(0.1)
    network.plot_random_walk(p.chain)
    print(p.calc_energy(p.chain))
