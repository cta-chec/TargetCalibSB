from TargetCalibSB import get_cell_ids_for_waveform
from TargetCalibSB.stats import welfords_online_algorithm
from TargetCalibSB.tf.base import TFAbstract
import numpy as np
from numba import njit, prange
import fitsio
from os.path import exists
from os import remove


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
        print(f"Saving TF tcal file: {path}")
        if exists(path):
            remove(path)
        shape = (self.shape[0], np.prod(self.shape[1:]))

        header = dict(
            TYPE=4,
            TM=self.shape[0] // 64,
            PIX=64,
            CELLS=self.shape[1],
            PNTS=self.shape[2],
        )
        with fitsio.FITS(path, 'rw') as file:
            file.create_image_hdu()
            file[0].write_keys(header)
            file.write(dict(CELLS=self.tf.reshape(shape)), extname="DATA")
            file.write(dict(CELLS=self.hits.reshape(shape)), extname="HITS")
            file.write(dict(CELLS=self._input_amplitudes), extname="AMPLITUDES")

    def load_tcal(self, path):
        print(f"Loading TF tcal file: {path}")
        with fitsio.FITS(path) as file:
            try:
                tf = file["DATA"].read()["CELLS"].reshape(self.shape)
                hits = file["HITS"].read()["CELLS"].reshape(self.shape)
                self._tf = tf.astype(np.float32)
                self._hits = hits.astype(np.float32)
                self._input_amplitudes = file["AMPLITUDES"].read()["CELLS"].astype(np.float64)
            except ValueError:
                raise ValueError("Incompatible TF class for file")

            self._amplitude_lookup = dict(zip(
                self._input_amplitudes,
                range(self._input_amplitudes.size)
            ))
            self._apply_amplitudes = None

    @classmethod
    def from_tcal(cls, path):
        with fitsio.FITS(path) as file:
            header = file[0].read_header()
            n_pixels = int(header['TM'] * header['PIX'])
            n_cells = int(header['CELLS'])
            n_amplitudes = int(header['PNTS'])

        instance = cls(n_pixels, 128, 4096, [0])
        instance.shape = (n_pixels, n_cells, n_amplitudes)
        instance._tf = np.zeros(instance.shape, dtype=np.float32)
        instance._hits = np.zeros(instance.shape, dtype=np.uint32)
        instance._m2 = np.zeros(instance.shape, dtype=np.float32)
        instance.load_tcal(path)
        return instance
