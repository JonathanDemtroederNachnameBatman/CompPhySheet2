import numpy as np
from matplotlib import pyplot as plt
from numba import jit, int32, boolean, int8, void, float64, optional
from numba.experimental import jitclass

@jitclass([('chain', int8[:,:]), ('grid', int8[:,:]), ('folds', int32), ('actual_folds', int32), ('J', float64[:, :]), ('energy', float64)])
class Protein:

    def __init__(self, chain, J):
        # protein chain (self avoiding random walk)
        self.chain = chain
        # quadratic grid of protein where each cell is the index of an amino acid of the chain # TODO resize if necessary
        self.grid = create_chain_grid(self.chain) # 2d grid with chain indexes
        self.folds = 0 # fold attempts, also increases when the fold is rejected due to higher energy
        self.actual_folds = 0 # actual amount of folds
        self.J = J
        self.energy = self.calc_energy(self.chain)

    def calc_size(self):
        # returns euclidian distance between Protein endpoints
        x0 = self.chain[0][0]
        y0 = self.chain[0][1]

        x1 = self.chain[-1][0]
        y1 = self.chain[-1][1]

        return np.sqrt((x1-x0)**2 + (y1-y0)**2)

    def is_foldable(self, amino_acid):
        return self.fold_step(amino_acid, True)

    def is_foldable_at(self, x, y):
        a = self.grid[x][y]
        return False if a < 0 else self.is_foldable(self.chain[a])

    def get_amino_acid(self, x, y, chain=None):
        # returns amino acid at position (x, y)
        # optional: pass a specific chain-array to not edit the protein attribute
        if chain is None:
            chain = self.chain
        a = self.grid[x][y]
        if a < 0: return np.full(7, -1, dtype=np.int8)
        return chain[a]

    def __fold_endpoint(self, amino_acid, test, d, temperature):
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
            shift = self.check_bounds(x, y)
            x += shift[0]
            y += shift[1]
            if self.grid[x][y] < 0: break
            # remove spot and try again
            possible_dir = np.delete(possible_dir, np.where(possible_dir == d0)[0])
        if test: return True
        # found spot in direction d0 -> fold
        index = self.grid[amino_acid[0]][amino_acid[1]]
        chain_copy = self.chain.copy()
        chain_copy = self.__execute_endpoint_fold(chain_copy, d[0], d0, x, y, index, x0, y0)
        self.__check_accept_fold(chain_copy, amino_acid[0], amino_acid[1], x, y, index, temperature)
        return True

    def check_bounds(self, x, y):
        # it can happen that the protein wanders towards a grid border
        # checks if a position is out of grid border
        if x < 0 or y < 0 or x >= self.grid.shape[0] or y >= self.grid.shape[1]:
            #print(f'Folding point is out of bounds after {self.folds}!')
            return self.center_chain()
        return np.zeros(2, dtype=np.int8)

    def center_chain(self):
        # recenters the acid chain and returns the x,y values the chain was shifted by
        x0 = self.chain[0][0]
        y0 = self.chain[0][1]
        optimise_chain(self.chain)
        self.grid = create_chain_grid(self.chain)
        return np.array([self.chain[0][0] - x0, self.chain[0][1] - y0], dtype=np.int8)

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

    def __execute_fold(self, chain, directions, x_new, y_new, amino_acid_index):

        amino_acid = chain[amino_acid_index]

        a = self.get_amino_acid(move_x(amino_acid[0], directions[0]), move_y(amino_acid[1], directions[0]), chain)
        b = self.get_amino_acid(move_x(amino_acid[0], directions[1]), move_y(amino_acid[1], directions[1]), chain)

        amino_acid[0] = x_new
        amino_acid[1] = y_new

        # remove old connection
        amino_acid[directions[0] + 1] = 0
        amino_acid[directions[1] + 1] = 0
        # set connection to opposite than before (True for all cases)
        amino_acid[opposite(directions[0]) + 1] = opposite(directions[0])
        amino_acid[opposite(directions[1]) + 1] = opposite(directions[1])
        # neighbors can be tricky to think about
        a[opposite(directions[0]) + 1] = 0
        a[directions[1] + 1] = directions[1]  # sieht komisch aus, ist aber so
        b[opposite(directions[1]) + 1] = 0
        b[directions[0] + 1] = directions[0]
        return chain

    def __execute_endpoint_fold(self, chain, old_dir, new_dir, x_new, y_new, amino_acid_index, x0, y0):
        amino_acid = chain[amino_acid_index]
        amino_acid[0] = x_new
        amino_acid[1] = y_new
        amino_acid[old_dir + 1] = 0
        amino_acid[opposite(new_dir) + 1] = opposite(new_dir)
        b = self.get_amino_acid(x0, y0, chain)
        b[opposite(old_dir) + 1] = 0
        b[new_dir + 1] = new_dir
        return chain

    def __check_accept_fold(self, chain, x_old, y_old, x_new, y_new, amino_acid_index, temperature):
        self.folds += 1
        energy_new = self.calc_energy(chain)
        delta_energy = energy_new - self.energy
        if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / temperature):
            self.grid[x_old][y_old] = -1
            self.grid[x_new][y_new] = amino_acid_index
            self.chain = chain
            self.energy = energy_new
            self.actual_folds += 1
            return True # fold accepted
        return False


    def fold_step(self, amino_acid, test, temperature=1):
        """
        Tries to fold an amino acid in a protein chain
        :param temperature: protein temperature to evaluate transition probability in monte carlo step
        :param amino_acid: an entry of a random walk list
        :param test: True if folding should not happen, but rather only check if folding is possible
        :return: if amino acid was folded successfully (or can be folded if test is True)
        """

        c = amino_acid[2:6]
        d = np.where(c > 0)[0] + 1 # connected directions
        if len(d) == 0 or len(d) > 2: return False  # invalid
        if (c[0] == 1 and c[2] == 1) or (c[1] == 1 and c[3] == 1): return False  # straight lines can't fold
        if len(d) == 1: # endpoint
            return self.__fold_endpoint(amino_acid, test, d, temperature)

        # move in both directions for x and y
        x = move_x(0, d[0]) + move_x(0, d[1]) + amino_acid[0]
        y = move_y(0, d[0]) + move_y(0, d[1]) + amino_acid[1]
        shift = self.check_bounds(x, y)
        x += shift[0]
        y += shift[1]
        if self.grid[x][y] < 0:  # spot is empty
            if test: return True # fold possible, stop here if test

            index = self.grid[amino_acid[0]][amino_acid[1]]
            chain_copy = self.chain.copy()

            chain_copy = self.__execute_fold(chain=chain_copy,
                                             directions=d,
                                             x_new=x,
                                             y_new=y,
                                             amino_acid_index=index)
            self.__check_accept_fold(chain_copy, amino_acid[0], amino_acid[1], x, y, index, temperature)
            return True # fold successful
        return False # fold unsuccessful

    def random_fold_step(self, temperature=1):
        # folds a random amino acid
        # tries again if it failed
        options = np.arange(len(self.chain), dtype=np.int8)
        while True:
            if len(options) == 0:
               # print('Protein is unfoldable')
                return False
            if len(options) == 1:
                i = options[0]
            else:
                i = np.random.choice(options)
            acid = self.chain[i]
            if self.fold_step(acid, test=False, temperature=temperature):
                return True
            options = np.delete(options, np.where(options == i)[0])

    def verify_chain_grid(self):
        # check if amino acid position matches with grid
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                a = self.grid[i][j]
                if a >= 0:
                    c = self.chain[a]
                    if c[0] != i or c[1] != j:
                       # print('Chain and grid are invalid')
                        return False
        return True

    def calc_energy(self, chain=None):
        if chain is None:
            chain = self.chain
        energy = 0
        for i in range(len(chain)):
            x0 = chain[i][0]
            y0 = chain[i][1]
            for j in range(len(chain)):
                if not (i-1 <= j <= i+1):
                    x1 = chain[j][0]
                    y1 = chain[j][1]
                    delta = abs(x1 - x0) + abs(y1 - y0)

                    if delta == 1:
                        energy += self.J[chain[i][6]][chain[j][6]] / 2 # correct for double-counts

        return energy

    def calc_next_neighbors(self, chain=None):
        if chain is None:
            chain = self.chain
        n = 0
        for i in range(len(chain)):
            x0 = chain[i][0]
            y0 = chain[i][1]
            for j in range(len(chain)):
                if not (i-1 <= j <= i+1):
                    x1 = chain[j][0]
                    y1 = chain[j][1]
                    delta = abs(x1 - x0) + abs(y1 - y0)

                    if delta == 1:
                        n += 1 / 2 # correct for double-counts

        return n

    def eigenvals_J(self):
        return np.linalg.eigvals(self.J + 0j) # add imaginary part to compensate for possible (forbidden) domain changes


@jit()
def create_protein(interaction_type='normal'):
    if interaction_type == 'const':
        J = const_interaction(20, -3, random_sign=False)
    elif interaction_type == 'const_random_sign':
        J = const_interaction(20, -3, random_sign=True)
    elif interaction_type == 'normal':
        # sigma needs to be chosen like this to cancel out the 2
        J = random_interaction(20, -3, 1/np.sqrt(2))
    else:
        raise Exception('Invalid interaction type. Valid values are ["const", "const_random_sign", "normal"].')
    chain = random_walk(30, True, True, True)
    return Protein(chain, J)


@jit('float64[:,:](int32,float64,boolean)')
def const_interaction(size, value, random_sign):
    J = np.full((size, size), value)
    if random_sign:
        choices = np.array([True, False])
        for i in range(J.shape[0]):
            for j in range(i, J.shape[1]):
                if np.random.choice(choices):
                    J[i][j] *= -1
                    J[j][i] = J[i][j] # keep symmetric
    return J

@jit(float64[:,:](int32,float64,float64))
def random_interaction(size, mean, sigma):
    # https://numpy.org/doc/2.0/reference/random/generated/numpy.random.normal.html
    J = np.random.normal(mean, sigma, size=(size, size))
    # symmetric
    for i in range(J.shape[0]):
        for j in range(i+1, J.shape[1]):
            J[j][i] = J[i][j]
    return J

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

@jit(int8[:,:](int8[:,:]))
def create_chain_grid(chain):
    s = len(chain) + 2
    grid = np.full((s, s), -1, dtype=np.int8) # chain is guaranteed to always fit
    # calc rectangular protein size
    xs = 0; ys = 0
    for c in chain:
        if c[0] > xs: xs = c[0]
        if c[1] > ys: ys = c[1]
    xs += 1; ys += 1
    x0 = s // 2 - xs // 2 # center pos
    y0 = s // 2 - ys // 2

    for i in range(len(chain)):
        c = chain[i]
        c[0] += x0 # offset each acid to center chain
        c[1] += y0
        grid[c[0]][c[1]] = i
    return grid

@jit(int8[:,:](int32,boolean,boolean,boolean))
def random_walk(length, optimise, self_avoiding, amino_acid):
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
               # print(f'Nowhere to run at {x0},{y0} at length {i}')
                return optimise_chain(kette[:i]) if optimise else kette[:i]
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
    return optimise_chain(kette) if optimise else kette

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

def plot_protein(protein: Protein, filename=''):
    fig, ax = plt.subplots(figsize=(10 / 2.54, 10 / 2.54))
    plt.grid()
    max_pos = protein.grid.shape[0]
    ax.set_xlim(-.5, max_pos+.5)
    ax.set_ylim(-.5, max_pos+.5)
    for i in range(1, len(protein.chain)):
        ax.plot([protein.chain[i - 1][0], protein.chain[i][0]], [protein.chain[i - 1][1], protein.chain[i][1]])

    size = round(protein.calc_size(), 2)
    plt.plot([], [], label='$r$ = '+str(size))
    plt.legend()
    if filename != '':
        plt.savefig(filename)
    else:
        plt.show()

# test functions
#print(opposite(1), opposite(2), opposite(3), opposite(4)) # 3, 4, 1, 2
#print(move_x(0, 1), move_x(0, 2), move_x(0, 3), move_x(0, 4)) # 0, 1, 0, -1
#print(move_y(0, 1), move_y(0, 2), move_y(0, 3), move_y(0, 4)) # 1, 0, -1, 0

#k = random_walk(100, True, False)
#plot_random_walk(k)
