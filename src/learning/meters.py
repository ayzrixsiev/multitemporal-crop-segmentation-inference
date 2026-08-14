"""
Drop-in replacement for torchnet.meter.AverageValueMeter.

Upstream's train_semantic.py does `import torchnet as tnt` and then uses exactly one
thing from that whole package: `tnt.meter.AverageValueMeter()`. torchnet has been
unmaintained since 2019 and does not install cleanly on modern Python, so the meter
lives here instead. Same three methods, same return contract:

    m = AverageValueMeter()
    m.add(loss.item())
    mean, std = m.value()      # train_semantic.py only ever reads value()[0]
    m.reset()

The running mean/std use Welford's online algorithm, so nothing is stored per step and
the numbers stay stable over thousands of batches. std is the sample (n-1) standard
deviation, matching torchnet. With zero samples both are nan; with one sample the mean
is that sample and std is inf, again matching torchnet.
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
