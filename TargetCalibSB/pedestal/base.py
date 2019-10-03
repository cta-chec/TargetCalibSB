from abc import abstractmethod
import numpy as np


class PedestalAbstract:
    def __init__(self, n_pixels, n_samples, n_cells):
        """
        Base class for a pedestal definition

        Use Welford's online algorithm (TargetCalibSB.stats.welfords) to
        iteratively build the pedestal

        Parameters
        ----------
        n_pixels : int
            Number of pixels
        n_samples : int
            Number of samples in a waveform
        n_cells : int
            Number of storage cells in the ASIC
        """
        self.shape = self.define_pedestal_dimensions(
            n_pixels, n_samples, n_cells
        )
        self._pedestal = np.zeros(self.shape, dtype=np.float32)
        self._hits = np.zeros(self.shape, dtype=np.uint32)
        self._m2 = np.zeros(self.shape, dtype=np.float32)

    def add_to_pedestal(self, wfs, fci):
        """
        Add samples from the waveforms from an event into the pedestal

        Parameters
        ----------
        wfs : ndarray
            Waveform array
            Shape: (n_pixels, n_samples)
        fci : int
            First cell id for the event
        """
        self._add_to_pedestal(wfs, fci, self._pedestal, self._hits, self._m2)

    def subtract_pedestal(self, wfs, fci):
        """
        Subtract the pedestal from the waveforms for an event

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
            Pedestal-subtracted waveforms
            Shape: (n_pixels, n_samples)
        """
        return self._subtract_pedestal(wfs, fci, self._pedestal)

    @property
    def pedestal(self):
        return self._pedestal

    @property
    def hits(self):
        return self._hits

    @property
    def variance(self):
        return self._m2 / (self._hits - 1)

    @property
    def std(self):
        return np.sqrt(self.variance)

    @staticmethod
    @abstractmethod
    def define_pedestal_dimensions(n_pixels, n_samples, n_cells):
        """
        Abstract method to define the shape of the pedestal arrays

        Parameters
        ----------
        n_pixels : int
            Number of pixels
        n_samples : int
            Number of samples in a waveform
        n_cells : int
            Number of storage cells in the ASIC

        Returns
        -------
        shape : tuple
        """

    @staticmethod
    @abstractmethod
    def _add_to_pedestal(wfs, fci, pedestal, hits, m2):
        """
        Method to define how this pedestal is built

        Parameters
        ----------
        wfs : ndarray
            Waveform array
            Shape: (n_pixels, n_samples)
        fci : int
            First cell id for the event
        pedestal : ndarray
            Storage array for the pedestal being built
            Shape: Defined in `define_pedestal_dimensions`
        hits : ndarray
            Storage array for number of hits for each pedestal entry
            Shape: Defined in `define_pedestal_dimensions`
        m2 : ndarray
            Storage array for the sum of squares of differences
            Shape: Defined in `define_pedestal_dimensions`
        """

    @staticmethod
    @abstractmethod
    def _subtract_pedestal(wfs, fci, pedestal):
        """
        Method to define how this pedestal is subtracted from the waveforms

        Parameters
        ----------
        wfs : ndarray
            Waveform array
            Shape: (n_pixels, n_samples)
        fci : int
            First cell id for the event
        pedestal : ndarray
            Pedestal array to subtract from samples
            Shape: Defined in `define_pedestal_dimensions`

        Returns
        -------
        subtracted : ndarray
            Pedestal-subtracted waveforms
            Shape: (n_pixels, n_samples)
        """
