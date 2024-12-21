import numpy as np
from matplotlib import pyplot as plt
from numba import jit, int32, bool, int8

# top = 1, right = 2, bottom = 3, left = 4

@jit
def opposite(dir):
    # maps directions to their opposite direction via smart modulo usage
    return dir % 4 + 2 if dir % 2 == 0 else (dir+1) % 4 + 1

@jit
def move_x(x, dir):
    # moves x pos in direction if direction is left or right
    return x - dir + 3 if dir % 2 == 0 else x

@jit
def move_y(y, dir):
    # moves y pos in direction if direction is up or down
    return y - dir + 2 if dir % 2 != 0 else y

@jit(bool(int8[:,:],int8,int8))
def contains(a, x, y):
    # return np.any(np.equal(a, [x, y]).all(1)) # for non jit
    for i in range(len(a)):
        if a[i][0] == x and a[i][1] == y:
            return True
    return False

@jit(int8[:,:](int32,bool))
def random_walk(length, self_avoiding):
    kette = np.zeros((length, 6), dtype=np.int8) # list of [x, y, top, right, bottom, left] where the last 4 values determine connections
    for i in range(1, length):
        directions = np.array([1, 2, 3, 4], dtype=np.int8)
        # previous position
        x0 = kette[i-1][0]
        y0 = kette[i-1][1]
        while True:
            if len(directions) == 0:
                print(f'Nowhere to run at {x0},{y0} at length {i}')
                return kette[:i]
            elif len(directions) == 1:
                # only 1 direction left to choose
                d = directions[0]
            else:
                # choose random direction to walk to
                d = np.random.choice(directions)
            # move position in direction
            x = move_x(x0, d)
            y = move_y(y0, d)
            # check if this is self avoiding and if the position already has been walked
            if not self_avoiding or not contains(kette, x, y):
                break
            # position already walked -> delete direction and choose again
            directions = np.delete(directions, np.where(directions == d)[0])
        #print(['up', 'right', 'down', 'left'][d-1])
        # update new position
        kette[i][0] = x
        kette[i][1] = y
        kette[i][opposite(d)+1] = 1 # set connection to previous pos
        kette[i-1][d+1] = 1 # set connection from previous pos to this pos
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
#print(opposite(1), opposite(2), opposite(3), opposite(4)) # 3, 4, 1, 2
#print(move_x(0, 1), move_x(0, 2), move_x(0, 3), move_x(0, 4)) # 0, 1, 0, -1
#print(move_y(0, 1), move_y(0, 2), move_y(0, 3), move_y(0, 4)) # 1, 0, -1, 0

k = random_walk(100, True)
plot_random_walk(k)
