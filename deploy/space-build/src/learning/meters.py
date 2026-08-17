"""
calculate average of loss after one epoch, batch
done to make it easy to look at during training
"""

import numpy as np


class AverageValueMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.mean = np.nan
        self.std = np.nan
        self._mean_old = 0.0
        self._m_s = 0.0

    def add(self, value, n=1):
        self.sum += value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum
            self.std = np.inf
            self._mean_old = self.mean
            self._m_s = 0.0
        else:
            self.mean = self._mean_old + (value - n * self._mean_old) / float(self.n)
            self._m_s += (value - self._mean_old) * (value - self.mean)
            self._mean_old = self.mean
            self.std = np.sqrt(self._m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std
