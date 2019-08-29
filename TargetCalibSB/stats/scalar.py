from TargetCalibSB.stats import welfords_online_algorithm
import numpy as np
from numba import njit


class OnlineStats:
    def __init__(self):
        self._mean = 0
        self._counts = 0
        self._m2 = 0

    def add_to_stats(self, wfs):
        self._mean, self._counts, self._m2 = self._add_to_stats(
            wfs, self._mean, self._counts, self._m2
        )

    @property
    def mean(self):
        return self._mean

    @property
    def counts(self):
        return self._counts

    @property
    def variance(self):
        return self._m2 / (self._counts - 1)

    @property
    def std(self):
        return np.sqrt(self.variance)

    @staticmethod
    @njit(fastmath=True)
    def _add_to_stats(wfs, mean, counts, m2):
        n_pixels, n_samples = wfs.shape
        for ipix in range(n_pixels):
            for isam in range(n_samples):
                sample = wfs[ipix, isam]
                mean, counts, m2 = welfords_online_algorithm(
                    sample, mean, counts, m2
                )
        return mean, counts, m2


class OnlineHist:
    def __init__(self, bins, range_):
        """
        Online building of a histgram for large datasets
        """
        self.hist, self.edges = np.histogram(
            [], bins=bins, range=range_,
        )

    def add(self, entries):
        self.hist += self._add(entries, self.edges)
        return self

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def _add(entries, edges):
        hist, _ = np.histogram(entries.ravel(), bins=edges)
        return hist
