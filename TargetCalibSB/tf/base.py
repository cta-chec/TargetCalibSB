from abc import abstractmethod
import numpy as np
from numba import guvectorize, float32, float64
import h5py
from os.path import exists
from os import remove
from datetime import datetime


@guvectorize([(float32[:], float64[:], float32[:])], '(s),(s)->(s)', nopython=True)
def make_monotonic(values, amplitudes, result):
    """
    Make an ndarray monotonically increasing over its last axis

    This function is a numpy universal function which defines the operation
    applied for every dimension.

    The result argument can be ignored when calling this function.
    It is filled inplace and returned by this function.

    Current method: Assume centre is well behaved. Iterate outward in both
    directions from centre and linearly interpolate values that don't follow
    the monotonically increasing trend. If there is no second value for
    interpolation (that is following the montonically increasing trend),
    then set to last behaved value.

    Parameters
    ----------
    values : ndarray
        The values to ensure are monotonic
        Shape: (*tf_dimensions, n_amplitudes)
    amplitudes : list or ndarray
       Amplitudes for the X axis of the TF
       Shape: (n_amplitudes)
    result : ndarray
        Return argument for ufunc (ignore)
        Returns the resulting monotonic array

    Returns
    -------
    result : ndarray
        Input values transformed to be monotonic
        Shape: (*tf_dimensions, n_amplitudes)
    """
    def interp(x, x0, x1, y0, y1):
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    result[:] = values
    n_entries = values.size
    mid = n_entries // 2
    # First half
    for i in range(mid, 0, -1):
        ii = i - 1
        if result[ii] < result[i]:
            continue
        else:
            for j in range(i, 0, -1):
                jj = j - 1
                if result[jj] < result[i]:
                    result[ii] = interp(
                        amplitudes[ii], amplitudes[i], amplitudes[jj],
                        result[i], result[jj]
                    )
                    break
            else:
                result[ii] = result[i]
    # Second half
    for i in range(mid, n_entries-1, 1):
        ii = i + 1
        if result[ii] > result[i]:
            continue
        else:
            for j in range(i, n_entries-1, 1):
                jj = j + 1
                if result[jj] > result[i]:
                    result[ii] = interp(
                        amplitudes[ii], amplitudes[i], amplitudes[jj],
                        result[i], result[jj]
                    )
                    break
            else:
                result[ii] = result[i]


@guvectorize([(float32[:], float64[:], float32[:])], '(s),(s)->(s)', nopython=True)
def get_zero_offset(tf, amplitudes, offset):
    """
    Get the offset for the amplitude axis that is required so that the
    interpolated value equals 0 if an input of 0 is provided

    This function is a numpy universal function which defines the operation
    applied for every dimension.

    The result argument can be ignored when calling this function.
    It is filled inplace and returned by this function.

    Parameters
    ----------
    tf : ndarray
        The TF values
        Shape: (*tf_dimensions, n_amplitudes)
    amplitudes : list or ndarray
       Amplitudes for the X axis of the TF
       Shape: (n_amplitudes)
    offset : ndarray
        Return argument for ufunc (ignore)

    Returns
    -------
    offset : ndarray
        Offest values for each TF entry
        Shape: (*tf_dimensions, n_amplitudes)
    """
    offset[:] = np.interp(0, tf, amplitudes)


class TFAbstract:
    def __init__(self, n_pixels, n_samples, n_cells, amplitudes):
        """
        Base class for a TF definition

        Use Welford's online algorithm (TargetCalibSB.stats.welfords) to
        iteratively build the TF

        Parameters
        ----------
        n_pixels : int
            Number of pixels
        n_samples : int
            Number of samples in a waveform
        n_cells : int
            Number of storage cells in the ASIC
        amplitudes : list or ndarray
           Amplitudes for each input file for the TF (1D)
        """
        n_amplitudes = len(amplitudes)
        self.shape = self.define_tf_dimensions(
            n_pixels, n_samples, n_cells, n_amplitudes
        )
        self._tf = np.zeros(self.shape, dtype=np.float32)
        self._hits = np.zeros(self.shape, dtype=np.uint32)
        self._m2 = np.zeros(self.shape, dtype=np.float32)
        # TODO: use float32 after https://github.com/numba/numba/issues/4890
        self._input_amplitudes = np.array(sorted(amplitudes), dtype=np.float64)
        self._amplitude_lookup = dict(zip(
            self._input_amplitudes,
            range(n_amplitudes)
        ))
        self._apply_amplitudes = None

    def get_input_amplitude_index(self, input_amplitude):
        return self._amplitude_lookup[input_amplitude]

    def finish_generation(self, vped_calibrator=None):
        if vped_calibrator:
            amplitudes = vped_calibrator.apply(self._input_amplitudes)
        else:
            amplitudes = np.tile(self._input_amplitudes, (self.shape[0], 1))

        # Convert input_amplitudes to the same dimensions as TF
        for i in range(self._tf.ndim - 2):
            amplitudes = np.expand_dims(amplitudes, axis=1)

        # Make TF monotonic
        tf = make_monotonic(self._tf, amplitudes)

        # Offset amplitudes
        offset = get_zero_offset(tf, amplitudes)
        amplitudes = amplitudes - offset

        self._apply_amplitudes = amplitudes
        self._tf = tf

    def add_to_tf(self, wfs, fci, amplitude_index):
        """
        Add samples from the waveforms from an event into the TF

        Parameters
        ----------
        wfs : ndarray
            Waveform array
            Shape: (n_pixels, n_samples)
        fci : int
            First cell id for the event
        amplitude_index : int
            Index for the amplitude axis
        """
        self._add_to_tf(wfs, fci, amplitude_index, self._tf, self._hits, self._m2)

    def apply_tf(self, wfs, fci):
        """
        Apply the TF to the waveforms for an event

        Parameters
        ----------
        wfs : ndarray
            Waveform array
            Shape: (n_pixels, n_samples)
        fci : int
            First cell id for the event

        Returns
        -------
        subtracted : ndarray
            Waveforms with TF applied
            Shape: (n_pixels, n_samples)
        """
        return self._apply_tf(wfs, fci, self._tf, self._apply_amplitudes)

    @property
    def x(self):
        if self._apply_amplitudes is None:
            raise ValueError("finish_generation has not been called for TF")
        return self._apply_amplitudes

    @property
    def tf(self):
        return self._tf

    @property
    def hits(self):
        return self._hits

    @property
    def variance(self):
        return self._m2 / (self._hits - 1)

    @property
    def std(self):
        return np.sqrt(self.variance)

    def save(self, path):
        print(f"Saving TF HDF5 file: {path}")
        if exists(path):
            remove(path)

        with h5py.File(path, "w") as file:
            file.create_dataset("TF", data=self._tf)
            file.create_dataset("HITS", data=self._hits)
            file.create_dataset("M2", data=self._m2)
            file.create_dataset("INPUT_AMPLITUDES", data=self._input_amplitudes)
            if self._apply_amplitudes is not None:
                file.create_dataset("APPLY_AMPLITUDES", data=self._apply_amplitudes)

            file.attrs["class_name"] = self.__class__.__name__
            file.attrs["datatime_created"] = str(datetime.utcnow())

    def load(self, path):
        print(f"Loading TF HDF5 file: {path}")
        with h5py.File(path, "r") as file:
            if len(self.shape) != file["TF"].ndim:
                raise ValueError("Incompatible TF class for file")
            self.shape = file["TF"].shape
            self._tf = file["TF"][:]
            self._hits = file["HITS"][:]
            self._m2 = file["M2"][:]
            self._input_amplitudes = file["INPUT_AMPLITUDES"][:]
            if "APPLY_AMPLITUDES" in file:
                self._apply_amplitudes = file["APPLY_AMPLITUDES"][:]

            self._amplitude_lookup = dict(zip(
                self._input_amplitudes,
                range(self._input_amplitudes.size)
            ))

    @classmethod
    def from_file(cls, path):
        instance = cls(64, 128, 4096, [0])
        instance.load(path)
        return instance

    @staticmethod
    @abstractmethod
    def define_tf_dimensions(n_pixels, n_samples, n_cells, n_amplitudes):
        """

        Parameters
        ----------
        n_pixels : int
            Number of pixels
        n_samples : int
            Number of samples in a waveform
        n_cells : int
            Number of storage cells in the ASIC
        n_amplitudes : int
            Number of amplitude entries for the TF

        Returns
        -------
        shape : tuple
        """

    @staticmethod
    @abstractmethod
    def _add_to_tf(wfs, fci, amplitude_index, tf, hits, m2):
        """
        Method to define how this TF is built

        Parameters
        ----------
        wfs : ndarray
            Waveform array
            Shape: (n_pixels, n_samples)
        fci : int
            First cell id for the event
        amplitude_index : int
            Lookup value in array for amplitude dimension
        tf : ndarray
            Storage array for the tf being built
            Shape: Defined in `define_tf_dimensions`
        hits : ndarray
            Storage array for number of hits for each tf entry
            Shape: Defined in `define_pedestal_dimensions`
        m2 : ndarray
            Storage array for the sum of squares of differences
            Shape: Defined in `define_pedestal_dimensions`
        """

    @staticmethod
    @abstractmethod
    def _apply_tf(wfs, fci, tf, amplitudes):
        """
        Method to define how this TF is applied to the waveforms

        Parameters
        ----------
        wfs : ndarray
            Waveform array
            Shape: (n_pixels, n_samples)
        fci : int
            First cell id for the event
        tf : ndarray
            TF array to apply to samples
            Shape: Defined in `define_tf_dimensions`
        amplitudes : ndarray
           Amplitudes for the X axis of the TF

        Returns
        -------
        subtracted : ndarray
            Waveforms with TF applied
            Shape: (n_pixels, n_samples)
        """
