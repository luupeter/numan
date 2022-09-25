import unittest
import tifffile as tif
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt

import vodex as vx
from numan.plots import *

# this folder has the processed folder results for 1v2v3v5 , 7 cycles, 94 volumes each
project_dir = "D:/Code/repos/numan/data/test/ld00_casper/processed"


class TestLabelPlotter(unittest.TestCase):
    experiment = vx.Experiment.load(Path(project_dir, "experiment_raw.db"))

    def test_init(self):
        lp = LabelPlotter(self.experiment, "number")
        names = ["b", "d1", "d2", "d3", "d5"]
        # labels in one cycle
        values = [0, 0, 0, 2,  # 0 - 3
                  0, 0, 0, 0, 0, 4,  # 4- 9
                  0, 0, 0, 0, 0, 0, 1,  # 10 - 16
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 2,  # 17 - 26
                  0, 0, 0, 0, 0, 4,  # 27 - 32
                  0, 0, 0, 0, 0, 0, 3,  # 33 - 39
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                  0, 0, 0, 0, 0, 3,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                  0, 0, 0, 0, 0, 0, 0, 3,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 4,
                  0, 0, 0, 0, 0, 0, 2,
                  0, 0, 0]

        self.assertListEqual(names, lp.names)
        self.assertListEqual(values, lp.values)

    def test_plot_labels(self):
        lp = LabelPlotter(self.experiment, "number")
        lp.plot_labels(show_plot=True)

    def test_plot_labels_forward_shift(self):
        lp = LabelPlotter(self.experiment, "number")
        lp.plot_labels(forward_shift=1, show_plot=True)

    def test_plot_labels_time_points(self):
        lp = LabelPlotter(self.experiment, "number")
        lp.plot_labels(time_points=[16, 3, 39, 9],
                       show_plot=True)

    def test_plot_labels_forward_shift_time_points(self):
        lp = LabelPlotter(self.experiment, "number")
        lp.plot_labels(forward_shift=1,
                       time_points=[16, 3, 39, 9],
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
        self.assertTupleEqual((7, 94), trace.shape)

    def test_plot_cycles(self):
        trace_id = 0
        ax_limits = self.s_plotter.plot_cycles(None, trace_id, show_plot=True)
        self.s_plotter.plot_cycles(None, trace_id, show_plot=True, forward_shift=3)
        self.assertAlmostEqual(
            [-0.5, 93.5, -0.004590296974461647, 0.039927868617393523],
            ax_limits)

    def test_prepare_psh(self):
        trace_id = 0
        labels = ["d1", "d2", "d3", "d5"]
        padding = [-2, -1, 0, 1, 2, 3, 4]
        trace = self.s_plotter.get_trace(trace_id)
        trace, label_selection = self.s_plotter.prepare_psh(trace, labels, padding)
        correct_label_selection = [14, 15, 16, 17, 18, 19, 20,
                                   1, 2, 3, 4, 5, 6, 7,
                                   37, 38, 39, 40, 41, 42, 43,
                                   7, 8, 9, 10, 11, 12, 13]
        self.assertListEqual(correct_label_selection, label_selection.tolist())

    def test_plot_psh_b(self):
        trace_id = 0
        labels = ["d1", "d2", "d3", "d5"]
        padding = [-2, -1, 0, 1, 2, 3, 4]
        _ = self.s_plotter.plot_psh(None, trace_id, labels, padding, show_plot=True)

    def test_plot_psh_0(self):
        trace_id = 0
        labels = ["d1", "d2", "d3", "d5"]
        padding = [0]
        _ = self.s_plotter.plot_psh(None, trace_id, labels, padding,
                                    split=False, plot_individual=False, show_plot=True)

    def test_make_psh_figure(self):
        trace_ids = [0, 1, 2, 3, 4]
        labels = ["d1", "d2", "d3", "d5"]
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


class TestSignalPlotterV2(unittest.TestCase):
    experiment = vx.Experiment.load(Path(project_dir, "experiment_w_covariates_raw.db"))
    spots = Spots.from_json(f"{project_dir}/spots/signals/spots_SvB_max.json")

    # initialise the signal plotter with DFF signal
    SLIDING_WINDOW = 15  # in volumes
    signals = spots.get_group_signals(spots.groups["sig2vB"]).as_dff(SLIDING_WINDOW)
    s_plotter_v2 = SignalPlotterV2(signals, experiment, "number")

    def test_get_labels_in_cycle(self):
        labels = ["d1"]
        padding = [0]
        selection = self.s_plotter_v2.get_labels_in_cycle(labels, padding=padding)
        self.assertListEqual(selection.tolist(), [[16], [49], [65]])

        labels = ["d1"]
        padding = [-1, 0, 1]
        selection = self.s_plotter_v2.get_labels_in_cycle(labels,
                                                          padding=padding, shift_by_padding=False)
        self.assertListEqual(selection.tolist(), [[15, 16, 17], [48, 49, 50], [64, 65, 66]])

    def test_prepare_psh(self):
        trace_id = 0
        labels = ["d1", "d2", "d3", "d5"]
        padding = [-2, -1, 0, 1, 2, 3, 4]
        trace = self.s_plotter_v2.get_trace(trace_id)
        trace, label_selection = self.s_plotter_v2.prepare_psh(trace, labels, padding)
        correct_label_selection = [14, 15, 16, 17, 18, 19, 20,
                                   1, 2, 3, 4, 5, 6, 7,
                                   37, 38, 39, 40, 41, 42, 43,
                                   7, 8, 9, 10, 11, 12, 13]
        self.assertListEqual(correct_label_selection, label_selection.tolist())

    def test_plot_psh_b(self):
        trace_id = 0
        labels = ["d1", "d2", "d3", "d5"]
        conditions = [
            [("number", "d1")],
            [("number", "d2")],
            [("number", "d3")],
            [("number", "d5")]
        ]
        padding = [-2, -1, 0, 1, 2, 3, 4]
        _ = self.s_plotter_v2.plot_psh(None, trace_id, labels, padding, show_plot=True)

    def test_plot_psh_0(self):
        trace_id = 0
        labels = ["d1", "d2", "d3", "d5"]
        padding = [0]
        _ = self.s_plotter_v2.plot_psh(None, trace_id, labels, padding,
                                       split=False, plot_individual=False, show_plot=True)

    def test_prepare_condition_trace(self):
        conditions = [
            [("number", "d1")],
            [("number", "d2")],
            [("number", "d3")],
            [("number", "d5")]
        ]
        padding = [-2, -1, 0, 1, 2, 3, 4]
        trace_id = 0
        # get the individual signal trace
        trace = self.s_plotter_v2.get_trace(trace_id)

        condition_signals = self.s_plotter_v2.prepare_condition_trace(trace,
                                                                      conditions,
                                                                      padding)
        self.assertTupleEqual(condition_signals.shape, (21, 28))

    def test_get_labels_for_conditions(self):
        conditions = [
            [("number", "d1")],
            [("number", "d2")],
            [("number", "d3")],
            [("number", "d5")]
        ]
        condition_label = ["d1", "d2", "d3", "d5"]
        cl = self.s_plotter_v2.get_labels_for_conditions(conditions)
        self.assertListEqual(condition_label, cl)

    def test_get_volumes_for_conditions_1c(self):
        conditions = [
            [("number", "d1")],
            [("number", "d2")],
            [("number", "d3")],
            [("number", "d5")]
        ]

        vols = self.s_plotter_v2.get_volumes_for_conditions(conditions)
        correct_vols = [[16, 49, 65, 110, 143, 159, 204, 237, 253, 298, 331,
                         347, 392, 425, 441, 486, 519, 535, 580, 613, 629],
                        [3, 26, 90, 97, 120, 184, 191, 214, 278, 285, 308, 372,
                         379, 402, 466, 473, 496, 560, 567, 590, 654],
                        [39, 55, 73, 133, 149, 167, 227, 243, 261, 321, 337, 355,
                         415, 431, 449, 509, 525, 543, 603, 619, 637],
                        [9, 32, 83, 103, 126, 177, 197, 220, 271, 291, 314, 365,
                         385, 408, 459, 479, 502, 553, 573, 596, 647]]
        self.assertEqual(4, len(vols))
        self.assertEqual(21, len(vols[0]))
        self.assertListEqual(vols, correct_vols)

    def test_get_volumes_for_conditions_2c(self):
        conditions = [
            [("number", "d1"), ("shape", "cr")],
            [("number", "d2"), ("shape", "cr")],
            [("number", "d3"), ("shape", "cr")],
            [("number", "d5"), ("shape", "cr")]
        ]

        vols = self.s_plotter_v2.get_volumes_for_conditions(conditions)
        correct_vols = [[16, 110, 204, 298, 392, 486, 580],
                        [3, 97, 191, 285, 379, 473, 567],
                        [39, 133, 227, 321, 415, 509, 603],
                        [9, 103, 197, 291, 385, 479, 573]]
        self.assertEqual(4, len(vols))
        self.assertEqual(7, len(vols[0]))
        self.assertListEqual(vols, correct_vols)

    def test_get_volumes_for_conditions_3c(self):
        conditions = [
            [("number", "d1"), ("shape", "cr"), ("spread", "ch")],
            [("number", "d2"), ("shape", "cr"), ("spread", "ch")],
            [("number", "d3"), ("shape", "cr"), ("spread", "ch")],
            [("number", "d5"), ("shape", "cr"), ("spread", "ch")]
        ]

        vols = self.s_plotter_v2.get_volumes_for_conditions(conditions)
        correct_vols = [[16, 204, 392, 580],
                        [3, 191, 379, 567],
                        [39, 227, 415, 603],
                        [9, 197, 385, 573]]
        self.assertEqual(4, len(vols))
        self.assertEqual(4, len(vols[0]))
        self.assertListEqual(vols, correct_vols)

    def test_plot_covariates_psh_b_same(self):
        trace_id = 0
        conditions = [
            [("number", "d1")],
            [("number", "d2")],
            [("number", "d3")],
            [("number", "d5")]
        ]
        padding = [-2, -1, 0, 1, 2, 3, 4]
        _ = self.s_plotter_v2.plot_covariates_psh(None, trace_id, conditions, padding, show_plot=True)

    def test_plot_covariates_psh_b_different(self):
        trace_id = 0
        conditions = [
            [("number", "d1"), ("shape", "cr")],
            [("number", "d2"), ("shape", "cr")],
            [("number", "d3"), ("shape", "cr")],
            [("number", "d5"), ("shape", "cr")]
        ]
        padding = [-2, -1, 0, 1, 2, 3, 4]
        _ = self.s_plotter_v2.plot_covariates_psh(None, trace_id, conditions, padding, show_plot=True)

    def test_plot_covariates_psh_b_different_no_individual(self):
        trace_id = 0
        conditions = [
            [("number", "d1"), ("shape", "cr")],
            [("number", "d2"), ("shape", "cr")],
            [("number", "d3"), ("shape", "cr")],
            [("number", "d5"), ("shape", "cr")]
        ]
        padding = [-2, -1, 0, 1, 2, 3, 4]
        _ = self.s_plotter_v2.plot_covariates_psh(None, trace_id, conditions, padding,
                                                  plot_individual=False, show_plot=True)

    def test_plot_covariates_psh_0_different(self):
        trace_id = 0
        conditions = [
            [("number", "d1"), ("shape", "cr")],
            [("number", "d2"), ("shape", "cr")],
            [("number", "d3"), ("shape", "cr")],
            [("number", "d5"), ("shape", "cr")]
        ]
        padding = [0]
        _ = self.s_plotter_v2.plot_covariates_psh(None, trace_id, conditions, padding,
                                                  split=False, plot_individual=False, show_plot=True)

    def test_prepare_cycle(self):
        trace_id = 0
        trace = self.s_plotter_v2.get_trace(trace_id)
        trace = self.s_plotter_v2.prepare_cycle(trace)
        self.assertTupleEqual((7, 94), trace.shape)

    def test_plot_cycles(self):
        trace_id = 0
        ax_limits = self.s_plotter_v2.plot_cycles(None, trace_id, show_plot=True)
        self.s_plotter_v2.plot_cycles(None, trace_id, show_plot=True, forward_shift=3)
        self.assertAlmostEqual(
            [-0.5, 93.5, -0.004590296974461647, 0.039927868617393523],
            ax_limits)

    def test_crop_and_pad(self):
        trace_id = 0
        trace = self.s_plotter_v2.get_trace(trace_id)
        vols = [0, 1, 2, 3]
        padding = [-2, -1, 0, 1, 2, 3]
        cropped_trace = self.s_plotter_v2.crop_and_pad(trace, vols, padding)
        correct_crop = [[-0.00010753, -0.00010753, -0.00010753, 0.00078853, 0.00120409, 0.00697],
                        [-0.00010753, -0.00010753, 0.00078853, 0.00120409, 0.00697, 0.01724216],
                        [-0.00010753, 0.00078853, 0.00120409, 0.00697, 0.01724216, 0.00376239],
                        [0.00078853, 0.00120409, 0.00697, 0.01724216, 0.00376239, -0.00060101]]
        np.testing.assert_almost_equal(cropped_trace, correct_crop, decimal=7)


if __name__ == "__main__":
    unittest.main()
