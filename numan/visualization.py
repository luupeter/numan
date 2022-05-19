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

def sort_by_len0(zip_to_sort):
    """
    Sorts the zip based on the length and alphabet of the first element in zip_to_sort
    """
    # sorts a list based on the length of the first element in zip
    sorted_zip = sorted(zip_to_sort, key=lambda x: (len(x[0])), reverse=True)
    return sorted_zip

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
                    ax.plot(cycled.T, noise_color, alpha=0.4, linewidth = 1)
                plot_errorbar(ax, mean, e)

            if vlines is not None:
                ax.vlines(vlines, ymin, ymax, linewidth=0.8, color='black')  # , linestyle=(0, (5, 10))

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

    def make_signal_reports(self, spot_tag, group_tag,
                     plot_type = "cycle",
                     plot_type_tag = '',
                     front_to_tail=0,
                     time_points=None,
                     vlines=None,
                     signal_split=None,
                     error_type="sem",
                     noise_color='--c',
                     plot_individual=False,
                     sort_by_sig = False):
        """
        plot_type: "cycle", "psh_b", "psh_0"
        """

        spots = Spots.from_json(f"{self.project}/spots/signals/spots_{spot_tag}.json")
        # where to temporary store images while the code is running
        tmp_folder = f"{self.project}/spots/reports/all_significant/signals/"
        # filename to save pdf with all the significant traces
        pdf_filename = f"{self.project}/spots/reports/all_significant/signals/" \
                       f"{plot_type}{plot_type_tag}_from_{spot_tag}_significance_{group_tag}.pdf"

        # initialise the signal plotter
        SLIDING_WINDOW = 15  # in volumes
        significant_signals_dff = spots.get_group_signals(spots.groups[group_tag]).as_dff(SLIDING_WINDOW)
        sp = SignalPlotter(significant_signals_dff, self.experiment)

        # some info on the cells to put into the title
        cells_idx = spots.get_group_idx(spots.groups[group_tag])
        cells_zyx = spots.get_group_centers(spots.groups[group_tag]).astype(np.int32)
        cells_group = spots.get_group_info(["sig2v3", "sig2v5", "sig3v5", "sig2vB", "sig3vB", "sig5vB"],
                                           group=spots.groups[group_tag])
        cells_group = np.array([group_name.replace("sig", "") for group_name in cells_group])
        signal_idx = np.arange(sp.n_signals)

        if sort_by_sig:
            # sort everything so that the cells with the most amount of significant stuff appear first
            sorted_zip = sort_by_len0(zip(cells_group, cells_idx, cells_zyx, signal_idx))
            cells_group = np.array([el[0] for el in sorted_zip])
            cells_idx = np.array([el[1] for el in sorted_zip])
            cells_zyx = np.array([el[2] for el in sorted_zip])
            signal_idx = np.array([el[3] for el in sorted_zip])

        main_title = f"DFF signals, tscore image {spot_tag}, significance {group_tag}"

        if plot_type == "psh_0":
            tpp = 10  # traces per page
        else:
            tpp = 5
        # prepare the batches per page
        cells = np.arange(sp.n_signals)
        btchs = [cells[s: s + tpp] for s in np.arange(np.ceil(sp.n_signals / tpp).astype(int)) * tpp]
        pdfs = []

        for ibtch, btch in enumerate(btchs):

            if plot_type == "cycle":
                # titles for the current batch
                titles = [f"Cell {idx}, {group} XYZ : {zyx[2]},{zyx[1]},{zyx[0]} (voxel) "
                          for idx, group, zyx in zip(cells_idx[btch], cells_group[btch], cells_zyx[btch])]

                sp.show_psh(signal_idx[btch],
                            main_title,
                            titles,
                            # front_to_tail will shift the cycleby the set number of voxels
                            # so when set to 3, there are 3 blank volumes at the begining and at the end ...
                            # if set to 0, will have 6 leading blanks and will end right after the 5 dots (black bar)
                            front_to_tail=front_to_tail,
                            # what grid to use to show the points
                            figure_layout=[5, 1],
                            # what error type to use ( "sem" for SEM or "prc" for 5th - 95th percentile )
                            error_type=error_type,
                            # figure parameters
                            figsize=(10, 12),
                            dpi=60,
                            # wheather to plot the individual traces
                            plot_individual=plot_individual,
                            # the color of the individual traces (if shown)
                            noise_color=noise_color)

            if plot_type == "psh_0":
                # titles for the current batch
                titles = [f"Cell {idx}, {group} \nXYZ : {zyx[2]},{zyx[1]},{zyx[0]} (voxel) "
                          for idx, group, zyx in zip(cells_idx[btch], cells_group[btch], cells_zyx[btch])]
                sp.show_psh(signal_idx[btch],
                            main_title,
                            titles,
                            # only show certain timepoints from the signal, for example : only 2 dots
                            time_points = time_points,
                            # front_to_tail will shift the cycleby the set number of voxels
                            # so when set to 3, there are 3 blank volumes at the begining and at the end ...
                            # if set to 0, will have 6 leading blanks and will end right after the 5 dots (black bar)
                            front_to_tail=0,
                            # what grid to use to show the points
                            figure_layout=[5, 2],
                            # what error type to use ( "sem" for SEM or "prc" for 5th - 95th percentile )
                            error_type=error_type,
                            # figure parameters
                            figsize=(10, 12),
                            dpi=60,
                            gridspec_kw={'hspace': 0.4, 'wspace': 0.3},
                            # wheather to plot the individual traces
                            plot_individual=plot_individual,
                            # the color of the individual traces (if shown)
                            noise_color=noise_color)

            if plot_type == "psh_b":
                # titles for the current batch
                titles = [f"Cell {idx}, {group} XYZ : {zyx[2]},{zyx[1]},{zyx[0]} (voxel) "
                          for idx, group, zyx in zip(cells_idx[btch], cells_group[btch], cells_zyx[btch])]
                sp.show_psh(signal_idx[btch],
                            main_title,
                            titles,
                            # only show certain timepoints from the signal, for example : only 2 dots
                            time_points=time_points,
                            # front_to_tail will shift the cycleby the set number of voxels
                            # so when set to 3, there are 3 blank volumes at the begining and at the end ...
                            # if set to 0, will have 6 leading blanks and will end right after the 5 dots (black bar)
                            front_to_tail=0,
                            # what grid to use to show the points
                            figure_layout=[5, 1],
                            # what error type to use ( "sem" for SEM or "prc" for 5th - 95th percentile )
                            error_type="sem",
                            # figure parameters
                            figsize=(10, 12),
                            dpi=60,
                            # if you wish to split the line
                            signal_split=signal_split,
                            # wheather to plot the individual traces
                            plot_individual=plot_individual,
                            # if you want to add vertical lines anywhere, list the x locations
                            vlines=vlines,
                            # the color of the individual traces (if shown)
                            noise_color=noise_color)

            plt.xlabel('Volume in cycle')
            filename = f'{tmp_folder}signals_batch{ibtch}.pdf'
            plt.savefig(filename)
            plt.close()
            pdfs.append(filename)

        merge_pdfs(pdfs, pdf_filename)
