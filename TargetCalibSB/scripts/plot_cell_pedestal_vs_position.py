import argparse
from argparse import ArgumentDefaultsHelpFormatter as Formatter
from TargetCalibSB.pedestal import PedestalCellPosition
from TargetCalibSB.stats import PixelStats, OnlineStats, OnlineHist
from CHECLabPy.core.io import TIOReader
from CHECLabPy.plotting.setup import Plotter
from CHECLabPy.plotting.camera import CameraImage
from CHECLabPy.utils.files import create_directory
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from os.path import join
from IPython import embed


class CellWaveform(Plotter):
    def plot(self, pedestal, std, hits, cell):
        n_samples = pedestal.size
        n_blocks = n_samples // 32
        block_end_samples = np.arange(n_blocks+1) * 32 + cell % 32

        for end_of_block in block_end_samples:
            start_of_block = end_of_block - 31
            color = self.ax._get_lines.get_next_color()
            if end_of_block >= n_samples:
                end_of_block = n_samples - 1
            self.ax.axvline(end_of_block, color=color, ls='--', alpha=0.7)
            if start_of_block < 0:
                start_of_block = 0
            self.ax.axvline(start_of_block, color=color, ls='--', alpha=0.7)

        x = np.where(hits > 0)[0]
        self.ax.errorbar(x, pedestal[x], yerr=std[x], color='black')

        self.ax.set_xlabel("Position in waveform")
        self.ax.set_ylabel("Amplitude (Raw ADC)")
        self.ax.set_title("Cell = {}".format(cell))

        self.ax.xaxis.set_major_locator(MultipleLocator(16))


def main():
    description = (
        "Generate the pedestals from an R0 file, subtract it from another "
        "R0 file, and plot the comparison of residuals from different "
        "pedestal methods"
    )
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=Formatter)
    parser.add_argument('-f', '--file', dest='r0_path', required=True,
                        help='R0 file to obtain residuals from')
    parser.add_argument('-o', '--output', dest='output_dir', required=True,
                        help='directort to store output plots')
    args = parser.parse_args()

    r0_path = args.r0_path
    channel = 7
    output_dir = args.output_dir

    create_directory(output_dir)
    reader = TIOReader(r0_path, max_events=100000)

    # Generate Pedestals
    pedestal = PedestalCellPosition(
        reader.n_pixels, reader.n_samples, reader.n_cells
    )
    desc = "Generating pedestal"
    for wfs in tqdm(reader, total=reader.n_events, desc=desc):
        if wfs.missing_packets:
            continue
        pedestal.add_to_pedestal(wfs, wfs.first_cell_id)

    # embed()

    for cell in range(703,reader.n_cells):
        p_cell_wf = CellWaveform()
        p_cell_wf.plot(
            pedestal.pedestal[channel, cell],
            pedestal.std[channel, cell],
            pedestal.hits[channel, cell],
            cell
        )
        p_cell_wf.save(
            join(output_dir, f"cell_pedestal_vs_position/{cell:04d}.pdf")
        )


if __name__ == '__main__':
    main()
