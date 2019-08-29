import argparse
from argparse import ArgumentDefaultsHelpFormatter as Formatter
from TargetCalibSB.pedestal import PedestalTargetCalib, PedestalBlockphase
from TargetCalibSB.stats import PixelStats, OnlineStats, OnlineHist
from CHECLabPy.core.io import TIOReader
from CHECLabPy.plotting.setup import Plotter
from CHECLabPy.plotting.camera import CameraImage
from CHECLabPy.utils.files import create_directory
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from os.path import join
from numba import njit, prange
from IPython import embed


class StatsPlot(Plotter):
    def plot(self, mean, std, label):
        pixel = np.arange(mean.size)

        (_, caps, _) = self.ax.errorbar(
            pixel, mean, yerr=std,
            fmt='o', mew=0.1, markersize=0.3, capsize=0.3, elinewidth=0.07,
            label=label
        )
        for cap in caps:
            cap.set_markeredgewidth(0.07)

    def finish(self):
        self.ax.set_xlabel("Pixel")
        self.ax.set_ylabel("Residuals (ADC)")
        self.add_legend("best")


class Camera2(Plotter):
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


class HistPlot(Plotter):
    def plot(self, hist, edges, mean, std, label):
        label += f"\nMean: {mean:.3f} StdDev: {std:.3f}"
        between = (edges[1:] + edges[:-1]) / 2
        self.ax.hist(
            between, bins=edges, weights=hist, label=label,
            histtype='step',
        )

    def finish(self):
        self.ax.set_xlabel("Residuals (ADC)")
        # self.ax.set_yscale("log")
        # self.ax.get_xaxis().set_major_formatter(
        #     FuncFormatter(lambda xl, _: '{:g}'.format(xl)))
        self.add_legend('best')


def main():
    # description = (
    #     "Generate the pedestals from an R0 file, subtract it from another "
    #     "R0 file, and plot the comparison of residuals from different "
    #     "pedestal methods"
    # )
    # parser = argparse.ArgumentParser(description=description,
    #                                  formatter_class=Formatter)
    # parser.add_argument('-f', '--file', dest='r0_path', required=True,
    #                     help='R0 file to obtain residuals from')
    # parser.add_argument('-p', '--pedestal', dest='pedestal_r0_path',
    #                     required=True,
    #                     help='R0 file to generate pedestal from')
    # parser.add_argument('-o', '--output', dest='output_dir', required=True,
    #                     help='directort to store output plots')
    # args = parser.parse_args()
    #
    # r0_path = args.r0_path
    # pedestal_r0_path = args.pedestal_r0_path
    # output_dir = args.output_dir

    r0_path = "/Users/Jason/Downloads/tempdata/Run06136_r0.tio"
    pedestal_r0_path = "/Users/Jason/Downloads/tempdata/Run06136_r0.tio"
    output_dir = "/Users/Jason/Downloads/compare_pedestals"

    create_directory(output_dir)
    reader_ped = TIOReader(pedestal_r0_path)  # TODO
    reader_res = TIOReader(r0_path, max_events=1000)  # TODO

    # Generate Pedestals
    pedestal_info = (
        reader_ped.n_pixels, reader_ped.n_samples, reader_ped.n_cells
    )
    pedestal_tc = PedestalTargetCalib(*pedestal_info)
    pedestal_bp = PedestalBlockphase(*pedestal_info)
    desc = "Generating pedestal"
    for wfs in tqdm(reader_ped, total=reader_ped.n_events, desc=desc):
        if wfs.missing_packets:
            continue
        pedestal_tc.add_to_pedestal(wfs, wfs.first_cell_id)
        pedestal_bp.add_to_pedestal(wfs, wfs.first_cell_id)

    pstats_tc = PixelStats(reader_res.n_pixels)
    pstats_bp = PixelStats(reader_res.n_pixels)
    stats_tc = OnlineStats()
    stats_bp = OnlineStats()
    hist_tc = OnlineHist(100, (-10, 10))
    hist_bp = OnlineHist(100, (-10, 10))

    # Subtract Pedestals
    desc = "Subtracting pedestal"
    for wfs in tqdm(reader_res, total=reader_res.n_events, desc=desc):
        if wfs.missing_packets:
            continue

        subtracted_tc = pedestal_tc.subtract_pedestal(wfs, wfs.first_cell_id)
        subtracted_bp = pedestal_bp.subtract_pedestal(wfs, wfs.first_cell_id)

        pstats_tc.add_to_stats(subtracted_tc)
        stats_tc.add_to_stats(subtracted_tc)
        hist_tc.add(subtracted_tc)

        pstats_bp.add_to_stats(subtracted_bp)
        stats_bp.add_to_stats(subtracted_bp)
        hist_bp.add(subtracted_bp)

    # Plot results
    label_tc = pedestal_tc.__class__.__name__
    label_bp = pedestal_bp.__class__.__name__

    p_pix_stats = StatsPlot()
    p_pix_stats.plot(pstats_tc.mean, pstats_tc.std, label_tc)
    p_pix_stats.plot(pstats_bp.mean, pstats_bp.std, label_bp)
    p_pix_stats.save(join(output_dir, "pix_stats.pdf"))

    p_ci_stats = Camera2(reader_res.mapping)
    p_ci_stats.set_image(pstats_tc.mean, pstats_tc.std)
    p_ci_stats.save(join(output_dir, f"ci_stats_{label_tc}.pdf"))
    p_ci_stats.set_image(pstats_bp.mean, pstats_bp.std)
    p_ci_stats.save(join(output_dir, f"ci_stats_{label_bp}.pdf"))

    p_hist = HistPlot()
    p_hist.plot(
        hist_tc.hist, hist_tc.edges, stats_tc.mean, stats_tc.std, label_tc
    )
    p_hist.plot(
        hist_bp.hist, hist_bp.edges, stats_bp.mean, stats_bp.std, label_bp
    )
    p_hist.save(join(output_dir, "hist.pdf"))


if __name__ == '__main__':
    main()
