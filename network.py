import random

import numpy as np
from matplotlib import pyplot as plt
from numba import jit

class Network:

    def __init__(self, size, length):
        self.size = size
        self.length = length
        self.network = np.zeros((self.size+2, self.size+2))

    def random_walk(self):
        for i in range(self.length):
            pass


# top = 1, right = 2, bottom = 3, left = 4

@jit
def opposite(dir):
    # maps directions to their opposite direction via smart modulo usage
    return dir % 4 + 2 if dir % 2 == 0 else (dir+1) % 4 + 1

@jit
def move_x(x, dir):
    return x - dir + 3 if dir % 2 == 0 else x

@jit
def move_y(y, dir):
    return y - dir + 2 if dir % 2 != 0 else y

#@jit # errors
def random_walk(length, self_avoiding=True):
    kette = np.zeros((length, 6)) # list of [x, y, top, right, bottom, left] where the last 4 values determine connections
    walked = np.zeros((length, 2))
    for i in range(1, length):
        directions = np.array([1, 2, 3, 4])
        x0 = kette[i-1][0]
        y0 = kette[i-1][1]
        while True:
            if len(directions) == 0:
                print(f'Nowhere to run at {x0},{y0} at length {i}')
                return kette[:i]
            elif len(directions) == 1:
                d = directions[0]
            else:
                d = np.random.choice(directions)
            x = move_x(x0, d)
            y = move_y(y0, d)
            if not self_avoiding or not np.any(np.equal(walked, [x,y]).all(1)):
                break
            directions = np.delete(directions, np.where(directions == d)[0])
        #print(['up', 'right', 'down', 'left'][d-1])
        kette[i][0] = x
        kette[i][1] = y
        walked[i][0] = x
        walked[i][1] = y
        kette[i][d+1] = 1
        kette[i-1][opposite(d)+1] = 1
    return kette

def plot_random_walk(kette):
    fig, ax = plt.subplots(figsize=(10/2.54, 10/2.54))
    plt.grid()
    max_pos = np.max(kette, axis=0)
    min_pos = np.min(kette, axis=0)
    ax.set_xlim(min(min_pos[0], min_pos[1])-.1, max(max_pos[0], max_pos[1])+.1)
    ax.set_ylim(min(min_pos[0], min_pos[1])-.1, max(max_pos[0], max_pos[1])+.1)
    for i in range(1, len(kette)):
        ax.plot([kette[i-1][0], kette[i][0]], [kette[i-1][1], kette[i][1]])
    plt.show()

# test functions
#print(opposite(1), opposite(2), opposite(3), opposite(4))
#print(move_x(0, 1), move_x(0, 2), move_x(0, 3), move_x(0, 4))
#print(move_y(0, 1), move_y(0, 2), move_y(0, 3), move_y(0, 4))

k = random_walk(100)
#print(k)
plot_random_walk(k)