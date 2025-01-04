import numpy as np
import matplotlib.pyplot as plt
import network
from numba import jit, njit, double, int8, int64
from numba_progress import ProgressBar


@njit(nogil=True)
def monte_carlo_temperature(protein, temperature_0, temperature_1, number_of_steps, number_of_repeats, progress_proxy):
    temperature = np.linspace(temperature_0, temperature_1, number_of_steps)
    simulation = np.zeros(shape=(number_of_steps, 3), dtype=double)

    for step in range(number_of_steps):
        energy_mean = 0
        size_mean = 0
        n = 0
        n_squared = 0
        for temperature_step in range(number_of_repeats):
            protein.random_fold_step(temperature[step])
            energy_mean += protein.calc_energy()
            size_mean += protein.calc_size()
            n += protein.calc_next_neighbors()
            n_squared += n ** 2
            progress_proxy.update(1)

        n /= number_of_repeats
        n_squared /= number_of_repeats

        energy_mean /= number_of_repeats
        size_mean /= number_of_repeats

        simulation[step][0] = energy_mean
        simulation[step][1] = size_mean
        simulation[step][2] = 0.3 * (n_squared - n ** 2) / (temperature[step] ** 2)

    return simulation


steps = 500
repeats = 10000000
num_iterations = steps * repeats
T_0 = 10
T_1 = 1
p = network.create_protein(interaction_type='const')
with ProgressBar(total=num_iterations) as progress:
    arr_T = monte_carlo_temperature(p, T_0, T_1, steps, repeats, progress)
energy = arr_T[:, 0]
size = arr_T[:, 1]
capacity = arr_T[:, 2]

fig, ax = plt.subplots()

ax.set_xlabel('Temperatur in a.u.')
ax.set_ylabel('Energie in a.u.')
plt.grid()

ax.plot(np.linspace(T_0, T_1, int(steps)), energy, color='blue')
plt.savefig('ex6-energy-1e3-1e8')

fig, ax = plt.subplots()

ax.set_xlabel('Temperatur in a.u.')
ax.set_ylabel('Abstand in a.u.')
plt.grid()

ax.plot(np.linspace(T_0, T_1, int(steps)), size, color='blue')
plt.savefig('ex6-size-1e3-1e8')

fig, ax = plt.subplots()

ax.set_xlabel('Temperatur in a.u.')
ax.set_ylabel('Abstand in a.u.')
plt.grid()

ax.plot(np.linspace(T_0, T_1, int(steps)), capacity, color='blue')
plt.savefig('ex6-capacity-1e3-1e8')