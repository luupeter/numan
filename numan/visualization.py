from tifffile import TiffFile, imread, imsave
import numpy as np
import json
import os

import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import pandas as pd
import PyPDF2

from .analysis import *
from .utils import *

class SignalPlotter:
    """
    All the plotting functions.
    """
    def __init__(self, signals, experiment, spf=1):
        """
        spf : seconds per frame
        """
        self.signals = signals
        self.experiment = experiment
        self.n_signals = self.signals.traces.shape[1]

    def plot_labels(self, ax, extent=None, time_points=None, front_to_tail=None):
        # timing in volumes, since one volume is one time point of the signal
        timing = (self.experiment.cycle.timing / self.experiment.volume_manager.fpv).astype(np.int)
        # get condition name for each time point of the signal
        conditions = [cond for t, condition in zip(timing, self.experiment.cycle.conditions) for cond in
                      [condition.name] * t]
        # encode unique names into intengers, return_inverse gives the integer encoding
        names, values = np.unique(conditions, return_inverse=True)

        if time_points is not None:
            time_points = np.array(time_points)
            time_shape = time_points.shape
            assert len(time_shape) < 3, "time_shape should be 1D or 2D"
            if len(time_shape) == 2:
                time_points = time_points[0, :]
            # take only the relevant part of the condition labels
            values = values[time_points]

        if front_to_tail is not None:
            old_order = np.arange(len(values))
            new_order = np.r_[old_order[front_to_tail:], old_order[0:front_to_tail]]
            values = values[new_order]

        img = ax.imshow(values[np.newaxis, :], aspect='auto',
                        extent=extent, cmap=plt.get_cmap('Greys', len(names)))
        img.set_clim(0, len(names) - 1)

        return names, values, img

    def show_labels(self, x_step=1):
        """
        Keep in mind - assign colors in alphabetic order of the condition name.
        """
        # TODO : for now it fits 3 different colors only! fix it!
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        names, values, img = self.plot_labels(ax)
        plt.xticks(ticks=np.arange(0, len(values), x_step))
        plt.xlabel('volume # per cycle')
        plt.title('Stimulus cycle')
        ax.get_yaxis().set_visible(False)

        # TODO : for now it fits 3 different colors only! fix it!
        cbar = plt.colorbar(img, ax=ax, ticks=[0.5, 1, 1.5], orientation='horizontal')
        cbar.ax.set_xticklabels(names)

    def show_psh(self, traces, main_title, tittles, error_type="prc", time_points=None,
                 plot_individual=True, front_to_tail=None,
                 figure_layout=None, figsize=None,
                 ylabel='', xlabel='', noise_color='--c', vlines=None, signal_split=None,
                 gridspec_kw=None,
                 dpi=160):
        """
        front_to_tail : how many cycle points to attach from front to tail
        """

        if figure_layout is not None:
            n_rows = figure_layout[0]
            n_col = figure_layout[1]
        else:
            n_rows = len(traces)
            n_col = 1

        if figsize is None:
            figsize = (12, n_rows * 4)

        fig, axes = plt.subplots(n_rows, n_col, gridspec_kw=gridspec_kw, figsize=figsize, dpi=dpi)
        axes = axes.flatten()
        fig.suptitle(main_title)
        for plot_id, trace in enumerate(traces):
            cycled, mean, e = self.signals.get_looped(trace, self.experiment, error_type=error_type,
                                                      time_points=time_points)

            if front_to_tail is not None:
                old_order = np.arange(len(mean))
                new_order = np.r_[old_order[front_to_tail:], old_order[0:front_to_tail]]

                cycled = cycled[:, new_order]
                mean = mean[new_order]
                e = e[:, new_order]

            ax = axes[plot_id]
            xmin, xmax, ymin, ymax = get_ax_limits(cycled, mean, e, plot_individual)
            names, _, img = self.plot_labels(ax, extent=[xmin, xmax, ymin, ymax],
                                             time_points=time_points,
                                             front_to_tail=front_to_tail)

            # if you wish to not connect certain groups of signals
            if signal_split is not None:
                for signal_group in signal_split:
                    if plot_individual:
                        ax.plot(signal_group, cycled[:, signal_group].T, noise_color, alpha=0.3)
                    plot_errorbar(ax, mean[signal_group], e[:, signal_group], x=signal_group)
            else:
                if plot_individual:
                    ax.plot(cycled.T, noise_color, alpha=0.3)
                plot_errorbar(ax, mean, e)

            if vlines is not None:
                ax.vlines(vlines, ymin, ymax, linewidth=0.2, color='black')  # , linestyle=(0, (5, 10))

            ax.set_title(tittles[plot_id])
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))
            ax.set_xticks(np.arange(len(mean)))
            ax.set_xticklabels([])
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)

        # cbar = plt.colorbar(img, ax=ax, ticks=[0.5, 1.5, 2.5], orientation='horizontal')
        # cbar.ax.set_xticklabels(names)


class Reports:
    """
    For now it is simply a class of wrappers to make reports specifically for the 2vs3vs5 experiment.
    I hope it will become more clean and general as time goes on.
    """
    def __init__(self, project_folder, experiment):
        self.project = project_folder
        self.experiment = experiment

    def make_reports_psh_0(self, spot_tag):
        spots = Spots.from_json(f"{self.project}/spots/signals/spots_{spot_tag}.json")

        group_tags = ["sig2v3", "sig2v5", "sig3v5", "sig2vB", "sig3vB", "sig5vB"]
        for group_tag in group_tags:

            # where to temporary store images while the cell is running
            tmp_folder = f"{self.project}/spots/reports/all_significant/signals/"
            # filename to save pdf with all the significant traces
            pdf_filename = f"{self.project}/spots/reports/all_significant/signals/" \
                           f"PSH_0_from_{spot_tag}_significance_{group_tag}.pdf"

            # initialise the signal plotter
            SLIDING_WINDOW = 15  # in volumes
            significant_signals_dff = spots.get_group_signals(spots.groups[group_tag]).as_dff(SLIDING_WINDOW)
            sp = SignalPlotter(significant_signals_dff, self.experiment)

            # some info on the cells to put into the title
            cells_idx = spots.get_group_idx(spots.groups[group_tag])
            cells_zyx = spots.get_group_centers(spots.groups[group_tag]).astype(np.int32)
            cells_group = spots.get_group_info(group_tags)[spots.groups[group_tag]]
            main_title = f"DFF signals, tscore image {spot_tag}, significance {group_tag}"

            # plotting parameters
            tpp = 10  # traces per page
            # prepare the batches per page
            cells = np.arange(sp.n_signals)
            btchs = [cells[s: s + tpp] for s in np.arange(np.ceil(sp.n_signals / tpp).astype(int)) * tpp]

            pdfs = []
            for ibtch, btch in enumerate(btchs):
                # titles for the current batch
                titles = [f"Cell {idx}, {group} \nXYZ : {zyx[2]},{zyx[1]},{zyx[0]} (voxel)"
                          for idx, group, zyx in zip(cells_idx[btch], cells_group[btch], cells_zyx[btch])]
                sp.show_psh(btch,
                            main_title,
                            titles,
                            # only show certain timepoints from the signal, for example : only 2 dots
                            time_points=[[13, 7, 20], [27, 37, 53], [43, 60, 70]],
                            # front_to_tail will shift the cycleby the set number of voxels
                            # so when set to 3, there are 3 blank volumes at the begining and at the end ...
                            # if set to 0, will have 6 leading blanks and will end right after the 5 dots (black bar)
                            front_to_tail=0,
                            # what grid to use to show the points
                            figure_layout=[5, 2],
                            # what error type to use ( "sem" for SEM or "prc" for 5th - 95th percentile )
                            error_type="sem",
                            # figure parameters
                            figsize=(10, 12),
                            dpi=60,
                            gridspec_kw={'hspace': 0.4, 'wspace': 0.3},
                            # wheather to plot the individual traces
                            plot_individual=False,
                            # the color of the individual traces (if shown)
                            noise_color='--c')

                plt.xlabel('Volume in cycle')
                filename = f'{tmp_folder}signals_batch{ibtch}.pdf'
                plt.savefig(filename)
                plt.close()
                pdfs.append(filename)

            merge_pdfs(pdfs, pdf_filename)

    def make_reports_cycle(self, spot_tag):
        spots = Spots.from_json(f"{self.project}/spots/signals/spots_{spot_tag}.json")

        group_tags = ["sig2v3", "sig2v5", "sig3v5", "sig2vB", "sig3vB", "sig5vB"]
        for group_tag in group_tags:

            # where to temporary store images while the cell is running
            tmp_folder = f"{self.project}/spots/reports/all_significant/signals/"
            # filename to save pdf with all the significant traces
            pdf_filename = f"{self.project}/spots/reports/all_significant/signals/" \
                           f"Cycles_from_{spot_tag}_significance_{group_tag}.pdf"

            # initialise the signal plotter
            SLIDING_WINDOW = 15  # in volumes
            significant_signals_dff = spots.get_group_signals(spots.groups[group_tag]).as_dff(SLIDING_WINDOW)
            sp = SignalPlotter(significant_signals_dff, self.experiment)

            # some info on the cells to put into the title
            cells_idx = spots.get_group_idx(spots.groups[group_tag])
            cells_zyx = spots.get_group_centers(spots.groups[group_tag]).astype(np.int32)
            cells_group = spots.get_group_info(group_tags)[spots.groups[group_tag]]
            main_title = f"DFF signals, tscore image {spot_tag}, significance {group_tag}"

            # plotting parameters
            tpp = 5  # traces per page
            # prepare the batches per page
            cells = np.arange(sp.n_signals)
            btchs = [cells[s: s + tpp] for s in np.arange(np.ceil(sp.n_signals / tpp).astype(int)) * tpp]

            pdfs = []
            for ibtch, btch in enumerate(btchs):
                # titles for the current batch
                titles = [f"Cell {idx}, {group} XYZ : {zyx[2]},{zyx[1]},{zyx[0]} (voxel) "
                          for idx, group, zyx in zip(cells_idx[btch], cells_group[btch], cells_zyx[btch])]
                sp.show_psh(btch,
                            main_title,
                            titles,
                            # front_to_tail will shift the cycleby the set number of voxels
                            # so when set to 3, there are 3 blank volumes at the begining and at the end ...
                            # if set to 0, will have 6 leading blanks and will end right after the 5 dots (black bar)
                            front_to_tail=3,
                            # what grid to use to show the points
                            figure_layout=[5, 1],
                            # what error type to use ( "sem" for SEM or "prc" for 5th - 95th percentile )
                            error_type="sem",
                            # figure parameters
                            figsize=(10, 12),
                            dpi=60,
                            # wheather to plot the individual traces
                            plot_individual=False,
                            # the color of the individual traces (if shown)
                            noise_color='--c')

                plt.xlabel('Volume in cycle')
                filename = f'{tmp_folder}signals_batch{ibtch}.pdf'
                plt.savefig(filename)
                plt.close()
                pdfs.append(filename)

            merge_pdfs(pdfs, pdf_filename)
