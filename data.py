import numpy as np
import math


class DataLoader:
    def __init__(self, X, y=None, batch_size=64, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(X)

    def __len__(self):
        return math.ceil(self.num_samples / self.batch_size)

    def __iter__(self):
        idx = np.random.permutation(self.num_samples) if self.shuffle else np.arange(self.num_samples)
        for start in range(0, self.num_samples, self.batch_size):
            batch_idx = idx[start:start + self.batch_size]
            if self.y is not None:
                yield self.X[batch_idx], self.y[batch_idx]
            else:
                yield self.X[batch_idx]
