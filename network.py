import numpy as np
from matplotlib import pyplot as plt
from numba import jit, int32, boolean, int8, void
from numba.experimental import jitclass


@jitclass([('chain', int8[:,:]), ('grid', int8[:,:]), ('foldable', int8[:,:])])
class Protein:

    def __init__(self):
        # protein chain (self avoiding random walk)
        self.chain = random_walk(30, True, True)
        # quadratic grid of protein where each cell is the index of an amino acid of the chain # TODO resize if necessary
        self.grid = create_chain_grid(self.chain) # 2d grid with chain indexes
        # list of foldable points # TODO might remove this
        self.foldable = np.full((0, 2), -1, dtype=np.int8)
        self.check_expand(-1,-1)
        self.check_expand(-1,-1)
        self.check_expand(self.grid.shape[0], self.grid.shape[1])
        self.check_expand(self.grid.shape[0], self.grid.shape[1])

    def is_foldable(self, amino_acid):
        return self.fold_step(amino_acid, True)

    def is_foldable_at(self, x, y):
        a = self.grid[x][y]
        return False if a < 0 else self.is_foldable(self.chain[a])

    def get_amino_acid(self, x, y):
        a = self.grid[x][y]
        if a < 0: return np.full(7, -1, dtype=np.int8)
        return self.chain[a]

    def find_foldable(self):
        for i in range(len(self.chain)):
            if self.is_foldable(self.chain[i]):
                self.foldable[i][0] = self.chain[i][0]
                self.foldable[i][1] = self.chain[i][1]
            else:
                self.foldable[i][0] = -1
                self.foldable[i][1] = -1

    def _fold_endpoint(self, amino_acid, test, d):
        # tries to fold an endpoint of a protein chain
        # internal use only
        # USE fold_step() instead

        x0 = move_x(amino_acid[0], d[0])
        y0 = move_y(amino_acid[1], d[0])
        b = self.get_amino_acid(x0, y0)
        # all directions without an existing connection
        possible_dir = other_dir(b[2:6])
        while True:
            if len(possible_dir) == 0: return False # all possible spots obstructed
            if len(possible_dir) == 1:
                d0 = possible_dir[0]
            else:
                d0 = np.random.choice(possible_dir) # select random spot
            x = move_x(x0, d0)
            y = move_y(y0, d0)
            shift = self.check_expand(x, y)
            x += shift[0]
            y += shift[1]
            if self.grid[x][y] < 0: break
            # remove spot and try again
            possible_dir = np.delete(possible_dir, np.where(possible_dir == d0)[0])
        if test: return True
        # found spot in direction d0 -> fold
        index = self.grid[amino_acid[0]][amino_acid[1]]
        self.grid[amino_acid[0]][amino_acid[1]] = -1
        self.grid[x][y] = index
        amino_acid[0] = x
        amino_acid[1] = y
        amino_acid[d[0]+1] = 0
        amino_acid[opposite(d0) + 1] = opposite(d0)
        b[opposite(d[0])+1] = 0
        b[d0+1] = d0
        return True

    def check_expand(self, x, y):
        p = np.array([0,0], dtype=np.int8)
        if x < 0:
            p += self.expand_grid(4, 2)
        elif x >= self.grid.shape[0]:
            p += self.expand_grid(2, 2)
        if y < 0:
            p += self.expand_grid(1, 2)
        elif y >= self.grid.shape[1]:
            p += self.expand_grid(3, 2)
        return p

    def expand_grid(self, direction, amount):
        x = 0
        y = 0
        if direction % 2 == 0:
            expand = np.full((amount, self.grid.shape[1]), -1, dtype=np.int8)
            if direction == 2:
                self.grid = np.vstack((self.grid, expand))
            else:
                self.grid = np.vstack((expand, self.grid))
                x = amount
        else:
            expand = np.full((self.grid.shape[0], amount), -1, dtype=np.int8)
            if direction == 2:
                self.grid = np.hstack((self.grid, expand))
            else:
                self.grid = np.hstack((expand, self.grid))
                y = amount
        if x > 0 or y > 0:
            for i in range(len(self.chain)):
                self.chain[i][0] += x
                self.chain[i][1] += y
        return np.array([x, y], dtype=np.int8)


    def fold_step_at(self, x, y, test):
        """
        Tries to fold an amino acid in a protein chain
        :param x: x pos of the acid in the grid
        :param y: y pos of the grid in the acid
        :param test: True if folding should not happen, but rather only check if folding is possible
        :return: if amino acid was folded successfully (or can be folded if test is True)
        """
        a = self.grid[x][y]
        return False if a < 0 else self.fold_step(self.chain[a], test)

    def fold_step(self, amino_acid, test):
        """
        Tries to fold an amino acid in a protein chain
        :param amino_acid: an entry of a random walk list
        :param test: True if folding should not happen, but rather only check if folding is possible
        :return: if amino acid was folded successfully (or can be folded if test is True)
        """

        c = amino_acid[2:6]
        d = np.where(c > 0)[0] + 1 # connected directions
        if len(d) == 0 or len(d) > 2: return False  # invalid
        if (c[0] == 1 and c[2] == 1) or (c[1] == 1 and c[3] == 1): return False  # straight lines can't fold
        if len(d) == 1: # endpoint
            return self._fold_endpoint(amino_acid, test, d)

        # move in both directions for x and y
        x = move_x(0, d[0]) + move_x(0, d[1]) + amino_acid[0]
        y = move_y(0, d[0]) + move_y(0, d[1]) + amino_acid[1]
        shift = self.check_expand(x, y)
        x += shift[0]
        y += shift[1]
        if self.grid[x][y] < 0:  # spot is empty
            if test: return True
            # neighboring acids
            a = self.get_amino_acid(move_x(amino_acid[0], d[0]), move_y(amino_acid[1], d[0]))
            b = self.get_amino_acid(move_x(amino_acid[0], d[1]), move_y(amino_acid[1], d[1]))
            # fold acid with connections in a right angle
            index = self.grid[amino_acid[0]][amino_acid[1]]
            self.grid[amino_acid[0]][amino_acid[1]] = -1
            self.grid[x][y] = index
            amino_acid[0] = x
            amino_acid[1] = y
            # remove old connection
            amino_acid[d[0]+1] = 0
            amino_acid[d[1]+1] = 0
            # set connection to opposite than before (True for all cases)
            amino_acid[opposite(d[0])+1] = opposite(d[0])
            amino_acid[opposite(d[1])+1] = opposite(d[1])
            # neighbors can be tricky to think about
            a[opposite(d[0])+1] = 0
            a[d[1]+1] = d[1] # sieht komisch aus, ist aber so
            b[opposite(d[1]) + 1] = 0
            b[d[0]+1] = d[0]
            return True
        return False

    def random_fold_step(self):
        options = np.arange(len(self.chain), dtype=np.int8)
        while True:
            if len(options) == 0:
                print('Protein is unfoldable')
                return False
            if len(options) == 1:
                i = options[0]
            else:
                i = np.random.choice(options)
            acid = self.chain[i]
            if self.fold_step(acid, test=False): return True
            options = np.delete(options, np.where(options == i)[0])


# top = 1, right = 2, bottom = 3, left = 4

@jit
def opposite(dir):
    # maps directions to their opposite direction via smart modulo usage
    return dir % 4 + 2 if dir % 2 == 0 else (dir+1) % 4 + 1

@jit
def other_dir(dir):
    all = [1,2,3,4]
    other = []
    for d in all:
        if d not in dir:
            other.append(d)
    return np.array(other, dtype=np.int8)

@jit
def move_x(x, dir):
    # moves x pos in direction if direction is left or right
    return x - dir + 3 if dir % 2 == 0 else x

@jit
def move_y(y, dir):
    # moves y pos in direction if direction is up or down
    return y - dir + 2 if dir % 2 != 0 else y

@jit(boolean(int8[:,:],int8,int8))
def contains(a, x, y):
    # return np.any(np.equal(a, [x, y]).all(1)) # for non jit
    for i in range(len(a)):
        if a[i][0] == x and a[i][1] == y:
            return True
    return False

@jit(int8[:,:](int8[:,:]))
def optimise_chain(chain):
    # shifts all position so that the smallest pos is 0,0
    x = 100000
    y = 100000
    for c in chain:
        if c[0] < x: x = c[0]
        if c[1] < y: y = c[1]
    for c in chain:
        c[0] -= x
        c[1] -= y
    return chain

@jit(int8(int8[:,:]))
def calc_quad_chain_size(chain):
    s = 0
    for c in chain:
        if c[0] > s: s = c[0]
        if c[1] > s: s = c[1]
    return s + 1

@jit(int8[:,:](int8[:,:]))
def create_chain_grid(chain):
    s = calc_quad_chain_size(chain)
    grid = np.full((s, s), -1, dtype=np.int8)
    for i in range(len(chain)):
        c = chain[i]
        grid[c[0]][c[1]] = i
    return grid

@jit(int8[:,:](int32,boolean,boolean))
def random_walk(length, self_avoiding, amino_acid):
    kette = np.zeros((length, 7 if amino_acid else 6), dtype=np.int8) # list of [x, y, top, right, bottom, left, amino_acid] where the last 4 values determine connections
    if amino_acid:
        kette[0][6] = np.random.randint(0, 20)
    for i in range(1, length):
        directions = np.array([1, 2, 3, 4], dtype=np.int8)
        # previous position
        x0 = kette[i-1][0]
        y0 = kette[i-1][1]
        while True:
            if len(directions) == 0:
                print(f'Nowhere to run at {x0},{y0} at length {i}')
                return optimise_chain(kette[:i])
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
        kette[i][opposite(d)+1] = opposite(d) # set connection to previous pos
        kette[i-1][d+1] = d # set connection from previous pos to this pos
        if amino_acid:
            kette[i][6] = np.random.randint(0, 20)
    return optimise_chain(kette)

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

#k = random_walk(100, True, False)
#plot_random_walk(k)
