from TargetCalibSB import get_cell_ids_for_waveform
from TargetCalibSB.stats import welfords_online_algorithm
from TargetCalibSB.tf.base import TFAbstract
from TargetCalibSB.tcal import load_tcal_tfinput, save_tcal_tfinput
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
        _, n_samples = wfs.shape
        _, n_cells, _ = tf.shape
        cells = get_cell_ids_for_waveform(fci, n_samples, n_cells)

        n_pixels, n_samples = wfs.shape
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
        _, n_samples = wfs.shape
        _, n_cells, _ = tf.shape
        cells = get_cell_ids_for_waveform(fci, n_samples, n_cells)

        n_pixels, n_samples = wfs.shape
        for ipix in prange(n_pixels):
            for isam in prange(n_samples):
                sample = wfs[ipix, isam]
                cell = cells[isam]
                tf_i = tf[ipix, cell]
                amplitudes_i = amplitudes[ipix, cell]
                calibrated[ipix, isam] = np.interp(sample, tf_i, amplitudes_i)
        return calibrated

    def save_tcal(self, path):
        save_tcal_tfinput(path, self.tf, self.hits, self._input_amplitudes)

    def load_tcal(self, path):
        tf, hits, input_amplitudes = load_tcal_tfinput(path)
        self._tf = tf.astype(np.float32)
        self._hits = hits.astype(np.float32)
        self._input_amplitudes = input_amplitudes.astype(np.float32)

        self._amplitude_lookup = dict(zip(
            self._input_amplitudes,
            range(self._input_amplitudes.size)
        ))
        self._apply_amplitudes = None

    @classmethod
    def from_tcal(cls, path):
        tf, hits, input_amplitudes = load_tcal_tfinput(path)
        n_pixels, n_cells, n_amplitudes = tf.shape
        instance = cls(n_pixels, 128, 4096, [0])
        instance.shape = (n_pixels, n_cells, n_amplitudes)  # Delayed because no wf
        instance._tf = tf.astype(np.float32)
        instance._hits = hits.astype(np.float32)
        instance._m2 = np.zeros(instance.shape, dtype=np.float32)
        instance._input_amplitudes = input_amplitudes.astype(np.float32)
        instance._amplitude_lookup = dict(zip(
            instance._input_amplitudes,
            range(instance._input_amplitudes.size)
        ))
        instance._apply_amplitudes = None
        return instance
