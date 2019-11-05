import argparse
from argparse import ArgumentDefaultsHelpFormatter as Formatter
from TargetCalibSB.pedestal import PedestalTargetCalib
from TargetCalibSB.stats import OnlineStats, OnlineHist
from CHECLabPy.core.io import TIOReader
from CHECLabPy.plotting.setup import Plotter
from CHECLabPy.utils.files import create_directory
from tqdm import tqdm
from os.path import join


class HistPlot(Plotter):
    def plot(self, hist, edges, mean, std):
        label = f"\nMean: {mean:.3f} StdDev: {std:.3f}"
        between = (edges[1:] + edges[:-1]) / 2
        self.ax.hist(
            between, bins=edges, weights=hist, label=label,
            histtype='step',
        )

    def finish(self):
        self.ax.set_xlabel("Residuals (ADC)")
        self.add_legend('best')


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
    pedestal = PedestalTargetCalib(
        reader_ped.n_pixels, reader_ped.n_samples, reader_ped.n_cells
    )
    desc = "Generating pedestal"
    for wfs in tqdm(reader_ped, total=reader_ped.n_events, desc=desc):
        if wfs.missing_packets:
            continue
        pedestal.add_to_pedestal(wfs, wfs.first_cell_id)

    online_stats = OnlineStats()
    online_hist = OnlineHist(bins=100, range_=(-10, 10))

    # Subtract Pedestals
    desc = "Subtracting pedestal"
    for wfs in tqdm(reader_res, total=reader_res.n_events, desc=desc):
        if wfs.missing_packets:
            continue

        subtracted_tc = pedestal.subtract_pedestal(wfs, wfs.first_cell_id)
        online_stats.add_to_stats(subtracted_tc)
        online_hist.add(subtracted_tc)

    p_hist = HistPlot()
    p_hist.plot(
        online_hist.hist, online_hist.edges, online_stats.mean, online_stats.std,
    )
    p_hist.save(join(output_dir, "hist.pdf"))


if __name__ == '__main__':
    main()
