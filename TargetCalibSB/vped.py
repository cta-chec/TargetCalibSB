import numpy as np
import pandas as pd
import h5py
from os.path import exists
from os import remove
from datetime import datetime
from numba import guvectorize, float64


@guvectorize(
    [(float64[:], float64[:], float64[:], float64[:])],
    '(a),(c),(c)->(a)',
    nopython=True
)
def _apply(vped, vpedf, voltagef, voltage_result):
    voltage_result[:] = np.interp(vped, vpedf, voltagef)


class VpedCalibrator:
    def __init__(self):
        self.vped = None
        self.voltage = None

    def apply(self, vped):
        """
        Apply the per-channel VPED calibration into mV

        Parameters
        ----------
        vped : ndarray
            VPED amplitudes
            Shape: (n_amplitudes)

        Returns
        -------
        mv : ndarray
            Corresponding amplitude in mV
            Shape: (n_channels, n_amplitudes)
        """
        return _apply(vped, self.vped, self.voltage)

    @staticmethod
    def _append_to_df(df_list, path, second_half):
        with open(path) as file:
            for iline, line in enumerate(file):
                if iline == 0:
                    continue

                data = line.replace('\n', '').split("\t")
                channel = np.arange(32)
                if second_half:
                    channel += 32
                df_list.append(pd.DataFrame(dict(
                    channel=channel,
                    vped=int(data[0]),
                    voltage=np.array(data[1:33], dtype=np.float),
                    temperature_primary=float(data[33]),
                    temperature_aux=float(data[34]),
                    temperature_power=float(data[35]),
                )))

    def read_ascii(self, channel_0_32_path, channel_32_64_path):
        """
        Read the VPED calibration from the ASCII files

        Parameters
        ----------
        channel_0_32_path : str
            Path to the ASCII file containing channels 0 - 32
        channel_32_64_path : str
            Path to the ASCII file containing channels 32 - 64
        """
        if channel_0_32_path == channel_32_64_path:
            raise ValueError("channel_0_32_path and channel_32_64_path are the same")

        df_list = []
        self._append_to_df(df_list, channel_0_32_path, False)
        self._append_to_df(df_list, channel_32_64_path, True)
        df = pd.concat(df_list, ignore_index=True)
        df = df.sort_values(["channel", "vped"])

        n_channels = np.unique(df['channel']).size
        n_vped = np.unique(df['vped']).size
        self.vped = df['vped'].values.reshape((n_channels, n_vped))
        self.voltage = df['voltage'].values.reshape((n_channels, n_vped))

    def save(self, path):
        print(f"Saving VPED calibration HDF5 file: {path}")
        if exists(path):
            remove(path)

        with h5py.File(path, "w") as file:
            file.create_dataset("VPED", data=self.vped)
            file.create_dataset("VOLTAGE", data=self.voltage)
            file.attrs["datatime_created"] = str(datetime.utcnow())

    def load(self, path):
        print(f"Loading VPED calibration HDF5 file: {path}")
        with h5py.File(path, "r") as file:
            self.vped = file["VPED"][:]
            self.voltage = file["VOLTAGE"][:]
