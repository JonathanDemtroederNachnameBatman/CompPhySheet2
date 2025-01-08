import numpy as np
import matplotlib.pyplot as plt
import network
from numba import njit, double
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


steps = 200
repeats = 12000000

T_0 = 10
T_1 = 1

num_of_proteins = 2
num_iterations = num_of_proteins * (steps * repeats - repeats)

p1 = network.create_protein(interaction_type='const')
p2 = network.Protein(np.copy(p1.chain), np.copy(p1.J))

with ProgressBar(total=num_iterations) as progress:
    arr_T_1 = monte_carlo_temperature(p1, T_0, T_1, steps, repeats, progress)
    arr_T_2 = monte_carlo_temperature(p2, T_0, T_1, steps, repeats, progress)

energy_1 = arr_T_1[:, 0]
size_1 = arr_T_1[:, 1]
capacity_1 = arr_T_1[:, 2]
energy_2 = arr_T_2[:, 0]
size_2 = arr_T_2[:, 1]
capacity_2 = arr_T_2[:, 2]

fig, ax = plt.subplots()

ax.set_xlabel('Temperatur in a.u.')
ax.set_ylabel('Energie in a.u.')
plt.grid()

ax.plot(np.linspace(T_0, T_1, int(steps)), energy_1, color='blue', label='Protein 1')
ax.plot(np.linspace(T_0, T_1, int(steps)), energy_1, color='red', label='Protein 2')
plt.legend(loc='lower right')
plt.savefig('pics/ex6/energy')

fig, ax = plt.subplots()

ax.set_xlabel('Temperatur in a.u.')
ax.set_ylabel('Abstand in a.u.')
plt.grid()

ax.plot(np.linspace(T_0, T_1, int(steps)), size_1, color='blue', label='Protein 1')
ax.plot(np.linspace(T_0, T_1, int(steps)), size_2, color='red', label='Protein 2')
plt.legend(loc='lower right')
plt.savefig('pics/ex6/size')

fig, ax = plt.subplots()

ax.set_xlabel('Temperatur in a.u.')
ax.set_ylabel('Abstand in a.u.')
plt.grid()

ax.plot(np.linspace(T_0, T_1, int(steps)), capacity_1, color='blue', label='Protein 1')
ax.plot(np.linspace(T_0, T_1, int(steps)), capacity_2, color='red', label='Protein 2')
plt.legend(loc='upper right')
plt.savefig('pics/ex6/capacity')

network.plot_protein(p1)
plt.savefig('pics/ex6/protein1')
network.plot_protein(p2)
plt.savefig('pics/ex6/protein2')