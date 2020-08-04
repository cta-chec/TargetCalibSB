from TargetCalibSB import get_cell_ids_for_waveform
from TargetCalibSB.stats import welfords_online_algorithm
from TargetCalibSB.tf.base import TFAbstract
import numpy as np
from numba import njit, prange


class TFDC(TFAbstract):
    """
    TF built from varying the DC voltage (VPED) applied to the waveforms
    """
    @staticmethod
    def define_tf_dimensions(n_pixels, n_samples, n_cells, n_amplitudes):
        shape = (n_pixels, n_cells, n_amplitudes)
        return shape

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def _add_to_tf(wfs, fci, amplitude_index, tf, hits, m2):
        n_pixels, n_samples = wfs.shape
        _, n_cells, _ = tf.shape
        cells = get_cell_ids_for_waveform(fci, n_samples, n_cells)

        for ipix in prange(n_pixels):
            for isam in prange(n_samples):
                sample = wfs[ipix, isam]
                idx = (ipix, cells[isam], amplitude_index)
                tf[idx], hits[idx], m2[idx] = welfords_online_algorithm(
                    sample, tf[idx], hits[idx], m2[idx]
                )

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def _apply_tf(wfs, fci, tf, amplitudes):
        calibrated = np.zeros(wfs.shape, dtype=np.float32)
        n_pixels, n_samples = wfs.shape
        _, n_cells, _ = tf.shape
        cells = get_cell_ids_for_waveform(fci, n_samples, n_cells)

        for ipix in prange(n_pixels):
            for isam in prange(n_samples):
                sample = wfs[ipix, isam]
                cell = cells[isam]
                tf_i = tf[ipix, cell]
                amplitudes_i = amplitudes[ipix, cell]
                calibrated[ipix, isam] = np.interp(sample, tf_i, amplitudes_i)
        return calibrated
