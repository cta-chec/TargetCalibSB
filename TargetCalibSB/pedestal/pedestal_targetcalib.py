from TargetCalibSB import N_BLOCKPHASE
from TargetCalibSB.stats import welfords_online_algorithm
from TargetCalibSB.pedestal.base import PedestalAbstract
import numpy as np
from numba import njit, prange
import fitsio
from os.path import exists
from os import remove


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
        print(f"Saving pedestal file: {path}")
        if exists(path):
            remove(path)
        shape = (self.shape[0], np.prod(self.shape[1:]))

        header = dict(
            TYPE=1,
            TM=self.shape[0] // 64,
            PIX=64,
            BLOCKS=self.shape[1],
            SAMPLESBP=self.shape[2],
        )
        pedestal = self._pedestal

        if compress:
            min_ = self._pedestal.min()
            max_ = self._pedestal.max()
            range_ = max_ - min_
            scale = int(65535 / (range_ + 1))
            if scale < 1:
                scale = 1
            offset = int(-1 * min_ + 1) * scale;
            header["SCALE"] = scale
            header["OFFSET"] = offset
            pedestal = np.round(pedestal * scale + offset).astype(np.uint16)

        with fitsio.FITS(path, 'rw') as file:
            file.create_image_hdu()
            file[0].write_keys(header)
            file.write(dict(CELLS=pedestal.reshape(shape)), extname="DATA")
            file.write(dict(CELLS=self.hits.reshape(shape)), extname="HITS")
            file.write(dict(CELLS=self.std.reshape(shape)), extname="STDDEV")

    def load_tcal(self, path):
        print(f"Loading pedestal file: {path}")
        with fitsio.FITS(path) as file:
            try:
                pedestal = file["DATA"].read()["CELLS"].reshape(self.shape)
                if pedestal.dtype == np.dtype('>u2'):
                    header = file[0].read_header()
                    scale = header['scale']
                    offset = header['offset']
                    pedestal = (pedestal.astype(np.float32) - offset) / scale
                self._pedestal = pedestal.astype(np.float32)
                if "HITS" and "STDDEV" in file:
                    hits = file["HITS"].read()["CELLS"].reshape(self.shape)
                    std = file["STDDEV"].read()["CELLS"].reshape(self.shape)
                    self._hits = hits
                    self._m2 = std ** 2 * (self._hits - 1)
            except ValueError:
                raise ValueError("Incompatible pedestal class for file")

    @classmethod
    def from_tcal(cls, path):
        with fitsio.FITS(path) as file:
            header = file[0].read_header()
            n_pixels = header['TM'] * header['PIX']
            n_blocks = header['BLOCKS']
            n_bpisam = header['SAMPLESBP']

        instance = cls(n_pixels, 128, 4096)
        instance.shape = (n_pixels, n_blocks, n_bpisam)
        instance._pedestal = np.zeros(instance.shape, dtype=np.float32)
        instance._hits = np.zeros(instance.shape, dtype=np.uint32)
        instance._m2 = np.zeros(instance.shape, dtype=np.float32)
        instance.load_tcal(path)
        return instance
