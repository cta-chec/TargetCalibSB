from TargetCalibSB import get_cell_ids_for_waveform
from TargetCalibSB.stats import welfords_online_algorithm
from TargetCalibSB.tf.tf_ac import TFAC
from TargetCalibSB.data import get_data
import numpy as np
from numba import njit, prange
from scipy.ndimage import correlate1d


class TFACCrossCorrelation(TFAC):
    """
    TF built from injecting electrical pulses onto the waveforms

    Amplitude of the pulses are extracted using the cross correlation technique
    """
    def __init__(self, n_pixels, n_samples, n_cells, amplitudes):
        super().__init__(n_pixels, n_samples, n_cells, amplitudes)

        template_path = get_data("pulse_template_average_wf.txt")
        template_x, template_y = np.loadtxt(template_path, unpack=True)
        template_x *= 1e9
        self.ref_x = np.arange(0, template_x.max() + 1, 1)
        self.ref_y = np.interp(self.ref_x, template_x, template_y)
        self.origin = self.ref_y.argmax() - self.ref_y.size // 2
        self.tmin = None
        self.tmax = None

    def set_trange(self, tmin, tmax):
        self.tmin = tmin
        self.tmax = tmax

    @staticmethod
    def _add_to_tf(wfs, fci, amplitude_index, tf, hits, m2):
        pass

    def extract_index(self, wfs, amplitude_index):
        amplitude = self._input_amplitudes[amplitude_index]
        if amplitude < 0:
            wfs = wfs * -1

        cc = correlate1d(wfs, self.ref_y, mode='constant', origin=self.origin)
        index = cc[:, self.tmin:self.tmax].argmax(axis=1) + self.tmin

        # from matplotlib import pyplot as plt
        # plt.cla()
        # plt.plot(wfs[0] / wfs[0].max())
        # plt.plot(cc[0] / cc[0].max())
        # plt.plot(self.ref_x - self.ref_y.argmax() + index[0], self.ref_y)
        # plt.axvline(index[0])
        # plt.pause(1)

        return index

    def add_to_tf(self, wfs, fci, amplitude_index):
        index = self.extract_index(wfs, amplitude_index)
        self._add_to_tf_cc(wfs, index, fci, amplitude_index, self._tf, self._hits, self._m2)

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def _add_to_tf_cc(wfs, index, fci, amplitude_index, tf, hits, m2):
        n_pixels, n_samples = wfs.shape
        _, n_cells, _ = tf.shape
        cells = get_cell_ids_for_waveform(fci, n_samples, n_cells)

        for ipix in prange(n_pixels):
            isam = index[ipix]
            sample = wfs[ipix, isam]
            idx = (ipix, cells[isam], amplitude_index)
            tf[idx], hits[idx], m2[idx] = welfords_online_algorithm(
                sample, tf[idx], hits[idx], m2[idx]
            )
