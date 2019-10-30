from TargetCalibSB import get_cell_ids_for_waveform
from TargetCalibSB.stats import welfords_online_algorithm
from TargetCalibSB.pedestal.base import PedestalAbstract
import numpy as np
from numba import njit, prange


class PedestalCellPosition(PedestalAbstract):
    """
    Method 2: per storage cell, per waveform position
    """
    @staticmethod
    def define_pedestal_dimensions(n_pixels, n_samples, n_cells):
        shape = (n_pixels, n_cells, n_samples)
        return shape

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def _add_to_pedestal(wfs, fci, pedestal, hits, m2):
        _, n_cells, n_samples = pedestal.shape
        cells = get_cell_ids_for_waveform(fci, n_samples, n_cells)

        n_pixels, n_samples = wfs.shape
        for ipix in prange(n_pixels):
            for isam in prange(n_samples):
                sample = wfs[ipix, isam]
                idx = (ipix, cells[isam], isam)
                pedestal[idx], hits[idx], m2[idx] = welfords_online_algorithm(
                    sample, pedestal[idx], hits[idx], m2[idx]
                )

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def _subtract_pedestal(wfs, fci, pedestal):
        subtracted = np.zeros(wfs.shape, dtype=np.float32)
        _, n_cells, n_samples = pedestal.shape
        cells = get_cell_ids_for_waveform(fci, n_samples, n_cells)

        n_pixels, n_samples = wfs.shape
        for ipix in prange(n_pixels):
            for isam in prange(n_samples):
                sample = wfs[ipix, isam]
                pedestal_value = pedestal[ipix, cells[isam], isam]
                if pedestal_value != 0:
                    subtracted[ipix, isam] = sample - pedestal_value
        return subtracted
