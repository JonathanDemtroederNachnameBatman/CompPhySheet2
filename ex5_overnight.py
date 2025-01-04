import numpy as np
import matplotlib.pyplot as plt
import network
from numba import jit, njit, double, int8, int64
from numba_progress import ProgressBar


@njit(nogil=True)
def monte_carlo_temperature(protein, temperature_0, temperature_1, number_of_steps, number_of_repeats, progress_proxy):
    temperature = np.linspace(temperature_0, temperature_1, number_of_steps)
    simulation = np.zeros(shape=(number_of_steps, 2), dtype=double)
    simulation[0][0] = protein.calc_energy()
    simulation[0][1] = protein.calc_size()

    for step in range(1, number_of_steps):
        energy_mean = 0
        size_mean = 0
        for temperature_step in range(number_of_repeats):
            protein.random_fold_step(temperature[step])
            energy_mean += protein.calc_energy()
            size_mean += protein.calc_size()
            progress_proxy.update(1)

        energy_mean /= number_of_repeats
        size_mean /= number_of_repeats

        simulation[step][0] = energy_mean
        simulation[step][1] = size_mean

    return simulation


steps = 100
repeats = 10000
num_iterations = 2* (steps * repeats - repeats)
T_0 = 10
T_1 = 1

# create two similar proteins
p = network.create_protein(interaction_type='normal')
p_copy = network.Protein(p.chain, p.J)


with ProgressBar(total=num_iterations) as progress:
    #np.random.seed(0)
    arr_T = monte_carlo_temperature(p, T_0, T_1, steps, repeats, progress)
    #np.random.seed(0)
    arr_T_copy = monte_carlo_temperature(p_copy, T_0, T_1, steps, repeats, progress)

energy = arr_T[:, 0]
size = arr_T[:, 1]
energy_copy = arr_T_copy[:, 0]
size_copy = arr_T_copy[:, 1]


fig, ax = plt.subplots()

ax.set_xlabel('Temperatur in a.u.')
ax.set_ylabel('Energie in a.u.')
plt.grid()

ax.plot(np.linspace(T_0, T_1, int(steps)), energy, color='blue', label='Protein 1')
ax.plot(np.linspace(T_0, T_1, int(steps)), energy_copy, color='red', label='Protein 2')
plt.legend(loc='lower right')
plt.savefig('ex5-energy')
