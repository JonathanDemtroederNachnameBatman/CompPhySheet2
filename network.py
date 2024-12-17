
import numpy as np

class Network:

    def __init__(self, size, length):
        self.size = size
        self.length = length
        self.network = np.zeros((self.size+2, self.size+2))

    def random_walk(self):
        for i in range(self.length):
            pass