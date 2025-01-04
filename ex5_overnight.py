import numpy as np
import matplotlib.pyplot as plt
import network
from numba import njit, double
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


steps = 200
repeats = 10000
num_iterations = 2 * (steps * repeats - repeats)
T_0 = 10
T_1 = 1

# create two similar proteins
p1 = network.create_protein(interaction_type='normal')
p2 = network.Protein(np.copy(p1.chain), np.copy(p1.J))

with ProgressBar(total=num_iterations) as progress:
    arr_T_1 = monte_carlo_temperature(p1, T_0, T_1, steps, repeats, progress)
    arr_T_2 = monte_carlo_temperature(p2, T_0, T_1, steps, repeats, progress)

energy_1 = arr_T_1[:, 0]
size_1 = arr_T_1[:, 1]
energy_2 = arr_T_2[:, 0]
size_2 = arr_T_2[:, 1]

# plot energy

fig, ax = plt.subplots()

ax.set_xlabel('Temperatur in a.u.')
ax.set_ylabel('Energie in a.u.')
plt.grid()

ax.plot(np.linspace(T_0, T_1, int(steps)), energy_1, color='blue', label='Protein 1')
ax.plot(np.linspace(T_0, T_1, int(steps)), energy_2, color='red', label='Protein 2')
plt.legend(loc='lower right')
plt.savefig('pics/ex5/energy')

# plot size

fig, ax = plt.subplots()

ax.set_xlabel('Temperatur in a.u.')
ax.set_ylabel('Abstand in a.u.')
plt.grid()

ax.plot(np.linspace(T_0, T_1, int(steps)), size_1, color='blue', label='Protein 1')
ax.plot(np.linspace(T_0, T_1, int(steps)), size_2, color='red', label='Protein 2')
plt.legend(loc='lower right')
plt.savefig('pics/ex5/size')

# plot proteins

network.plot_protein(p1)
plt.savefig('pics/ex5/protein1')
network.plot_protein(p2)
plt.savefig('pics/ex5/protein2')
