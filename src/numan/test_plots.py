import unittest
import tifffile as tif
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt

import vodex as vx
from plots import *

project_dir = "D:/Code/repos/numan/notebooks/data/v1.x.x/2vs3vs5/processed"


class TestLabelPlotter(unittest.TestCase):
    experiment = vx.Experiment.load(Path(project_dir, "experiment_raw.db"))

    def test_init(self):
        lp = LabelPlotter(self.experiment, "number")
        names = ["b", "d2", "d3", "d5"]
        values = [0, 0, 0, 0, 0, 0, 0, 2,  # 6,7 - 8
                  0, 0, 0, 0, 0, 1,  # 12,13 - 14
                  0, 0, 0, 0, 0, 0, 3,  # 19,20 - 21
                  0, 0, 0, 0, 0, 0, 1,  # 26,27 - 28
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 2,  # 36,37 - 38
                  0, 0, 0, 0, 0, 1,  # 42, 43 - 44
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 3,  # 52, 53 - 54
                  0, 0, 0, 0, 0, 0, 2,  # 59, 60 - 61
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 3]  # 69, 70
        self.assertListEqual(names, lp.names)
        self.assertListEqual(values, lp.values)

    def test_plot_labels(self):
        lp = LabelPlotter(self.experiment, "number")
        lp.plot_labels(show_plot=True)
        lp.plot_labels(forward_shift=1, show_plot=True)
        lp.plot_labels(time_points=[6, 7, 8, 12, 13, 14, 19, 20, 21],
                       show_plot=True)
        lp.plot_labels(forward_shift=1,
                       time_points=[6, 7, 8, 12, 13, 14, 19, 20, 21],
                       show_plot=True)


class TestSignalPlotter(unittest.TestCase):
    experiment = vx.Experiment.load(Path(project_dir, "experiment_raw.db"))
    spots = Spots.from_json(f"{project_dir}/spots/signals/spots_SvB_max.json")

    # initialise the signal plotter with DFF signal
    SLIDING_WINDOW = 15  # in volumes
    signals = spots.get_group_signals(spots.groups["sig2vB"]).as_dff(SLIDING_WINDOW)
    s_plotter = SignalPlotter(signals, experiment, "number")

    def test_prepare_cycle(self):
        trace_id = 0
        trace = self.s_plotter.get_trace(trace_id)
        trace = self.s_plotter.prepare_cycle(trace)
        self.assertTupleEqual((18, 71), trace.shape)

    def test_plot_cycles(self):
        trace_id = 0
        ax_limits = self.s_plotter.plot_cycles(None, trace_id, show_plot=True)
        self.s_plotter.plot_cycles(None, trace_id, show_plot=True, forward_shift=3)
        self.assertAlmostEqual(
            [-0.5, 70.5, -0.04479955511212163, 0.07627925625219258],
            ax_limits)

    def test_plot_psh(self):
        trace_id = 0
        labels = ["d2", "d3", "d5"]
        padding = [-2, -1, 0, 1, 2, 3, 4]
        _ = self.s_plotter.plot_psh(None, trace_id, labels, padding, show_plot=True)
        padding = [0]
        _ = self.s_plotter.plot_psh(None, trace_id, labels, padding,
                                    split=False, plot_individual=False, show_plot=True)

    def test_make_psh_figure(self):
        trace_ids = [0, 1, 2, 3, 4]
        labels = ["d2", "d3", "d5"]
        padding = [-2, -1, 0, 1, 2, 3, 4]
        _ = self.s_plotter.make_psh_figure(trace_ids, labels, padding,
                                           "Main title",
                                           ["0", "1", "2", "3", "4"],
                                           show_plot=True)

    def test_make_cycle_figure(self):
        trace_ids = [0, 1, 2, 3, 4]
        forward_shift = 3
        _ = self.s_plotter.make_cycle_figure(trace_ids,
                                             "Main title",
                                             ["0", "1", "2", "3", "4"],
                                             forward_shift=forward_shift,
                                             show_plot=True)

    def test_make_avg_act_scat_figure(self):
        labels = ["d2", "d3", "d5"]
        _ = self.s_plotter.make_avg_act_scat_figure(labels, "Title",
                                                    figure_layout=[1, 3],
                                                    figsize=(12, 10), dpi=160,
                                                    show_plot=True)
        labels = ["d2", "d3", "d5", "d2"]
        _ = self.s_plotter.make_avg_act_scat_figure(labels, "Title",
                                                    figure_layout=[2, 3],
                                                    figsize=(12, 10), dpi=160,
                                                    show_plot=True)


if __name__ == "__main__":
    unittest.main()
