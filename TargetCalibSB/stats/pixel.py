from TargetCalibSB.stats import welfords_online_algorithm
import numpy as np
from numba import njit, prange


class PixelStats:
    def __init__(self, n_pixels):
        self._mean = np.zeros(n_pixels, dtype=np.float64)
        self._counts = np.zeros(n_pixels, dtype=np.uint32)
        self._m2 = np.zeros(n_pixels, dtype=np.float64)

    def add_to_stats(self, wfs):
        self._add_to_stats(
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
    @njit(fastmath=True, parallel=True)
    def _add_to_stats(wfs, mean, counts, m2):
        n_pixels, n_samples = wfs.shape
        for ipix in prange(n_pixels):
            for isam in prange(n_samples):
                sample = wfs[ipix, isam]
                mean[ipix], counts[ipix], m2[ipix] = welfords_online_algorithm(
                    sample, mean[ipix], counts[ipix], m2[ipix]
                )
