import argparse
from argparse import ArgumentDefaultsHelpFormatter as Formatter
from TargetCalibSB.pedestal import PedestalSimple
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


class ChannelStd(Plotter):
    def plot(self, std):
        self.ax.plot(std)

    def finish(self):
        self.ax.set_xlabel("Pixel")
        self.ax.set_ylabel("Residuals StdDev (ADC)")
        self.add_legend("best")
        self.ax.set_ylim(top=3)


class CameraStats(Plotter):
    def __init__(self, mapping):
        super().__init__()

        self.fig = plt.figure(figsize=(8, 3))
        self.ax_mean = self.fig.add_subplot(1, 2, 1)
        self.ax_std = self.fig.add_subplot(1, 2, 2)
        self.ci_mean = CameraImage.from_mapping(mapping, ax=self.ax_mean)
        self.ci_std = CameraImage.from_mapping(mapping, ax=self.ax_std)
        self.ci_mean.add_colorbar("Residuals Mean (ADC)", pad=0.1)
        self.ci_std.add_colorbar("Residuals StdDev (ADC)", pad=0.1)

    def set_image(self, mean, std):
        self.ci_mean.image = mean
        self.ci_std.image = std


class Hist2D(Plotter):
    def plot(self, values, hits, clabel):
        masked = np.ma.masked_where(hits == 0, values)
        print(masked.max())
        im = self.ax.imshow(
            masked, cmap="viridis", origin='lower', aspect='auto'
        )
        cbar = self.fig.colorbar(im)
        self.ax.patch.set(hatch='xx')
        self.ax.set_xlabel("Cell ID")
        self.ax.set_ylabel("Channel")
        cbar.set_label(clabel)


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
    parser.add_argument('-p', '--pedestal', dest='pedestal_r0_path',
                        required=True,
                        help='R0 file to generate pedestal from')
    parser.add_argument('-o', '--output', dest='output_dir', required=True,
                        help='directort to store output plots')
    args = parser.parse_args()

    r0_path = args.r0_path
    pedestal_r0_path = args.pedestal_r0_path
    output_dir = args.output_dir

    create_directory(output_dir)
    reader_ped = TIOReader(pedestal_r0_path, max_events=100000)
    reader_res = TIOReader(r0_path, max_events=100000)

    # Generate Pedestals
    pedestal = PedestalSimple(
        reader_ped.n_pixels, reader_ped.n_samples, reader_ped.n_cells
    )
    desc = "Generating pedestal"
    for wfs in tqdm(reader_ped, total=reader_ped.n_events, desc=desc):
        if wfs.missing_packets:
            continue
        pedestal.add_to_pedestal(wfs, wfs.first_cell_id)

    channel_stats = PixelStats(reader_res.n_pixels)

    # Subtract Pedestals
    desc = "Subtracting pedestal"
    for wfs in tqdm(reader_res, total=reader_res.n_events, desc=desc):
        if wfs.missing_packets:
            continue

        subtracted_tc = pedestal.subtract_pedestal(wfs, wfs.first_cell_id)
        channel_stats.add_to_stats(subtracted_tc)

    embed()

    # Plot results
    p_channel_std = ChannelStd()
    p_channel_std.plot(channel_stats.std)
    p_channel_std.save(join(output_dir, "simple_channel_std.pdf"))

    p_ci_stats = CameraStats(reader_res.mapping)
    p_ci_stats.set_image(channel_stats.mean, channel_stats.std)
    p_ci_stats.save(join(output_dir, "simple_ci_stats.pdf"))

    p_hist2d_pedestal = Hist2D()
    p_hist2d_pedestal.plot(pedestal.pedestal, pedestal.hits, "Pedestal Mean (ADC)")
    p_hist2d_pedestal.save(join(output_dir, "simple_hist2d_mean.pdf"))

    p_hist2d_std = Hist2D()
    p_hist2d_std.plot(pedestal.std, pedestal.hits, "Pedestal Stddev (ADC)")
    p_hist2d_std.save(join(output_dir, "simple_hist2d_std.pdf"))


if __name__ == '__main__':
    main()
