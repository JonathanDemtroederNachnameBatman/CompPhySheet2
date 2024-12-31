from matplotlib import pyplot as plt

import network
import numpy as np
from numba import jit, double, int8, int64


@jit(double[:](int8, int64))
def monte_carlo_const_temperature(temperature, number_of_steps):
    simulation = np.zeros(number_of_steps, dtype=double)

    p = network.create_protein(interaction_type='normal')
    simulation[0] = p.energy
    for step in range(1, number_of_steps):
        p.random_fold_step(temperature)
        simulation[step] = p.energy

    return simulation

steps = 1e6
temperature = 1

arr = monte_carlo_const_temperature(temperature, steps)

fig, ax = plt.subplots()

ax.set_xlabel('Monte-Carlo-Schritte')
ax.set_ylabel('Energie in a.u.')
ax.plot(np.arange(steps), arr)
plt.savefig('test')