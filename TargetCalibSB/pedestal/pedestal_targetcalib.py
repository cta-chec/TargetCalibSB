from TargetCalibSB import N_BLOCKPHASE
from TargetCalibSB.stats import welfords_online_algorithm
from TargetCalibSB.pedestal.base import PedestalAbstract
from TargetCalibSB.tcal import save_tcal_pedestal, load_tcal_pedestal
import numpy as np
from numba import njit, prange


class PedestalTargetCalib(PedestalAbstract):
    """
    Method 0: per fblock, per fbpisam (TargetCalib equivalent)
    """
    @staticmethod
    def define_pedestal_dimensions(n_pixels, n_samples, n_cells):
        n_blocks = n_cells // N_BLOCKPHASE
        n_bpisam = N_BLOCKPHASE + n_samples - 1
        shape = (n_pixels, n_blocks, n_bpisam)
        return shape

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def _add_to_pedestal(wfs, fci, pedestal, hits, m2):
        fblockphase = fci % 32
        fblock = fci // 32

        n_pixels, n_samples = wfs.shape
        for ipix in prange(n_pixels):
            for isam in prange(n_samples):
                sample = wfs[ipix, isam]
                fbpisam = fblockphase + isam
                idx = (ipix, fblock, fbpisam)
                pedestal[idx], hits[idx], m2[idx] = welfords_online_algorithm(
                    sample, pedestal[idx], hits[idx], m2[idx]
                )

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def _subtract_pedestal(wfs, fci, pedestal):
        subtracted = np.zeros(wfs.shape, dtype=np.float32)
        fblockphase = fci % 32
        fblock = fci // 32

        n_pixels, n_samples = wfs.shape
        for ipix in prange(n_pixels):
            for isam in prange(n_samples):
                sample = wfs[ipix, isam]
                fbpisam = fblockphase + isam
                pedestal_value = pedestal[ipix, fblock, fbpisam]
                if pedestal_value != 0:
                    subtracted[ipix, isam] = sample - pedestal_value
        return subtracted

    def save_tcal(self, path, compress=False):
        save_tcal_pedestal(path, self._pedestal, self.hits, self.std, compress)

    def load_tcal(self, path):
        pedestal, hits, std = load_tcal_pedestal(path)
        self._pedestal = pedestal.astype(np.float32)
        self._hits = hits
        self._m2 = std ** 2 * (self._hits - 1)

    @classmethod
    def from_tcal(cls, path):
        pedestal, hits, std = load_tcal_pedestal(path)
        n_pixels, n_blocks, n_bpisam = pedestal.shape
        instance = cls(n_pixels, 128, 4096)
        instance.shape = (n_pixels, n_blocks, n_bpisam)  # Delayed because no wf
        instance._pedestal = pedestal.astype(np.float32)
        instance._hits = hits
        instance._m2 = std ** 2 * (hits - 1)
        return instance
