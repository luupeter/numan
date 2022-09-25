from tifffile import TiffFile, imread, imsave
import numpy as np
import json
import os

import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import pandas as pd

from itertools import combinations

import PyPDF2
from PyPDF2 import PdfFileReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import white, black
from scipy.stats import rankdata

# from .analysis import *
# from .utils import *
from numan import *


# def place_cb(can, x, y, name):
#     form = can.acroForm
#     can.setFont("Courier", 12)
#     can.drawCentredString(x + 20, y + 20, name)
#     form.checkbox(name=name,
#                   # tooltip = f"Field {name}",
#                   x=x + 10,
#                   y=y - 4,
#                   # buttonStyle = 'check',
#                   borderColor=black,
#                   fillColor=white,
#                   textColor=black,
#                   forceBorder=True
#                   )
#     return can
#

# def generate_timpoints(bb, ba, time_centers):
#     """
#     Adds the bb and ba number of blanks around each value in time_centers,
#     keeping the number of rows the same.
#     """
#     time_points = np.zeros((3, ((bb + ba) + 1) * 3))
#     for it, t in enumerate(time_centers):
#         a, b, c = t
#         a_long = np.concatenate((a - np.arange(bb + 1)[::-1], a + 1 + np.arange(ba)))
#         b_long = np.concatenate((b - np.arange(bb + 1)[::-1], b + 1 + np.arange(ba)))
#         c_long = np.concatenate((c - np.arange(bb + 1)[::-1], c + 1 + np.arange(ba)))
#         time_points[it] = np.concatenate((a_long, b_long, c_long))
#     time_points = time_points.astype(int)
#     return time_points


def shift_signal(values, forward_shift, axis=0):
    """
    Shifts signal along axis 0 by forward_shift number of units forward. ( Tail will be in the front )
    :param axis: 0 or -1 , whether to reorder the o dimension or the last one
    :type axis: int, 0/-1
    :param values: values to shift
    :type values: Union(list, numpy.array)
    :param forward_shift: how many units to shift forward by
    :type forward_shift: int
    :return: shifted signal
    :rtype: Union(list, numpy.array)
    """
    assert axis in [0, -1], f"axis should be 0 or -1, but {axis} was given"

    old_order = np.arange(values.shape[axis])
    new_order = np.r_[old_order[forward_shift:], old_order[0:forward_shift]]
    if axis == 0:
        values = values[new_order]
    elif axis == -1:
        values = values[:, new_order]
    return values


def select_columns(values, selection):
    """
    Selects only certain columns from the given array.
    :param values: array to select values from, can be 1d or 2d
    :type values: Union(list, numpy.array)
    :param selection: positions along axis 1 of the elements to grab
    :type selection: [int]
    :return: cropped
    :rtype: Union(list, numpy.array)
    """
    selection = np.array(selection)
    # returns 1d ("cropped") or 2d ("looped")
    selection_dim = len(selection.shape)
    values_dim = len(values.shape)

    if values_dim == 1:
        values = values[selection]

    # returns 2d ("cropped" or "looped")
    elif values_dim == 2:
        values = values[:, selection]
        if selection_dim == 2:
            values = values.reshape(-1, values.shape[2])

    return values


# def get_idx_per_page(spots, group_tag, sort_by_sig=False):
#     # some info on the cells to put into the title
#     cells_idx = spots.get_group_idx(spots.groups[group_tag])
#     if sort_by_sig:
#         cells_group = spots.get_group_info(["sig2v3", "sig2v5", "sig3v5", "sig2vB", "sig3vB", "sig5vB"],
#                                            group=spots.groups[group_tag])
#         cells_group = np.array([group_name.replace("sig", "") for group_name in cells_group])
#         # sort everything so that the cells with the most amount of significant stuff appear first
#         sorted_zip = sort_by_len0(zip(cells_group, cells_idx))
#         cells_group = np.array([el[0] for el in sorted_zip])
#         cells_idx = np.array([el[1] for el in sorted_zip])
#     tpp = 5
#     # prepare the batches per page
#     cells = np.arange(len(cells_idx))
#     btchs = [cells[s: s + tpp] for s in np.arange(np.ceil(len(cells_idx) / tpp).astype(int)) * tpp]
#     return cells_idx.astype(str), btchs


class LabelPlotter:
    """
    Plots conditions for one cycle or a portion of a cycle.
    """

    # TODO : add per frame possibility

    def __init__(self, experiment, annotation_type):
        """Only per volume labels are implemented at the moment"""
        self.experiment = experiment
        # get conditions per volumes
        self.names, self.values = self.conditions_per_cycle(annotation_type)

    def conditions_per_cycle(self, annotation_type):
        """
        Returns names of labels in one cycle and labels per volume for one cycle.
        :param annotation_type: the name of the annotation to list
        :type annotation_type: str
        :return: a list of label names that occur in one cycle
        and a list of label's ranks per volume in a cycle : [0,0,0,3, 0, 0 ...0, 1, 0, ..]
        :rtype: ([str],[int])
        """
        condition_ids, condition_names = self.experiment.list_conditions_per_cycle(annotation_type)
        # make it so that the conditions start at 0 and grow incrementally : 0, 1, 2, 3, ...
        conditions = np.unique(condition_ids)
        condition_rank = [np.where(conditions == c_id)[0][0] for c_id in condition_ids]
        # keep only the names of the conditions present:
        # some conditions might have been listed as possible conditions, but not used in annotation
        # (also numbering cond starts at 1, so "cond-1" )
        names = [condition_names[cond - 1] for cond in conditions]

        return names, condition_rank

    def plot_labels(self, ax=None, extent=None, time_points=None, forward_shift=None, show_plot=False):
        """
        Figures out how to plot labels.
        :param ax:
        :type ax:
        :param extent:
        :type extent:
        :param time_points: the columns from the data to keep
        :type time_points: Union([int], numpy.array(int))
        :param forward_shift:
        :type forward_shift:
        :param show_plot: whether to show the plotted axis
        :type show_plot: bool
        :return:
        :rtype:
        """
        if ax is None:
            ax = plt.gca()

        values = np.array(self.values)

        if time_points is not None:
            assert len(np.array(time_points).shape) == 1, f"time_points for labels have to be a 1d array," \
                                                          f"but got {time_points.shape}"
            # take only the relevant part of the condition labels
            values = select_columns(values, time_points)

        if forward_shift is not None:
            values = shift_signal(values, forward_shift, axis=0)

        img = ax.imshow(values[np.newaxis, :], aspect='auto',
                        extent=extent, cmap=plt.get_cmap('Greys', len(self.names)))
        img.set_clim(0, len(self.names) - 1)

        if show_plot:
            plt.show()

        return self.names, values, img


class SignalPlotter:
    """
    Plots signals per VOLUMES.
    """

    def __init__(self, signals, experiment, annotation_type,
                 mean_color='r', noise_color='-c',
                 # covariate mean and noise
                 c_mean_color='w', c_noise_color='-m', c_edge_color='w'):

        self.traces = signals.traces  # Time x N_traces
        self.n_traces = self.traces.shape[1]

        self.experiment = experiment
        self.annotation = annotation_type
        self.labels = LabelPlotter(experiment, annotation_type)

        # plotting parameters
        self.mean_color = mean_color
        self.noise_color = noise_color
        self.c_mean_color = c_mean_color
        self.c_noise_color = c_noise_color
        self.c_edge_color = c_edge_color

        self.error_type = "sem"

    def get_trace(self, trace_id):
        return self.traces[:, trace_id]

    def prepare_cycle(self, trace):
        """
        Preppares trace to be plotted as cycle.
        """
        cycle_iterations = self.experiment.list_cycle_iterations(self.annotation, as_volumes=True)
        cycles, volumes_per_cycle = np.unique(cycle_iterations, return_counts=True)
        trace = trace.reshape((len(cycles), volumes_per_cycle[0]))
        return trace

    @staticmethod
    def pad_cycle(trace, padding):
        pad_left = min(padding)
        # move everything down a cycle , so the end of cycle one appears at the beginning of cycle 2
        # NOTE: the end of the VERY last cycle appears at the very beginning of the recording
        trace_left = trace[0:-1, pad_left:]
        trace_left = np.vstack((trace[-1, pad_left:][None, :], trace_left))

        pad_right = max(padding)
        # move everything up a cycle , so the beginning of cycle 2 appears at the end of cycle 1
        # NOTE: the beginning of the VERY first cycle appears at the end of the recording
        trace_right = trace[1:, 0:pad_right]
        trace_right = np.vstack((trace_right, trace[0, 0:pad_right][None, :]))

        return np.hstack((trace_left, trace, trace_right))

    @staticmethod
    def crop_and_pad(trace, volumes, padding):
        """

        :param trace: signal trace from which to crop the time points
        :type trace: 1d numpy array
        :param volumes: list of central volumes to crop from the signal, of length V
        :type volumes: [int]
        :param padding: padding around the central volumes, of length P
        :type padding: [int]
        :return: cropped signal in a shape VxP
        :rtype: 2d numpy array
        """
        padding = np.array(padding)[np.newaxis, :]
        volumes = np.array(volumes)[:, np.newaxis]

        assert np.min(padding) <= 0, \
            f"Padding on the left should be negative or 0, but got {min(padding)}"
        assert np.max(padding) >= 0, \
            f"Padding on the right should be positive or 0, but got {max(padding)}"

        # will make a matrix VxP
        mask = volumes + padding
        # make sure you are not requesting things outside the trace
        # this will copy the values at the beginning/ end of the trace
        mask[mask < 0] = 0
        mask[mask >= len(trace)] = len(trace) - 1

        cropped_signal = trace[mask]
        return cropped_signal

    @staticmethod
    def get_trace_stats(trace, error_type="sem"):
        """
        Calculates trace statistics along axis = 0.
        Trace should be reshaped prior to calling this function.
        """
        mean = np.mean(trace, axis=0)
        if error_type == "prc":
            # error bars : 5 to 95 th percentile around the median
            e = np.r_[np.expand_dims(mean - np.percentile(trace, 5, axis=0), axis=0),
                      np.expand_dims(np.percentile(trace, 95, axis=0) - mean, axis=0)]
        elif error_type == "sem":
            # error bars : sem around hte mean
            sem = np.std(trace, axis=0, ddof=1) / np.sqrt(trace.shape[0])
            e = np.r_[np.expand_dims(sem, axis=0),
                      np.expand_dims(sem, axis=0)]
        else:
            e = None

        return mean, e

    def get_labels_in_cycle(self, labels, padding=[0], shift_by_padding=False):
        """
        :param shift_by_padding: Whether to shift the labels by the size of padding on the left.
        Used when the signal has had a padding added on the two sides.
        :type shift_by_padding: bool
        :param labels: list of the label names to grab, from the annotation
        :type labels: [str]
        :param padding: list of padding around each label's spot : negative to the left, positive to the right
        Example: [-2,-1,0,1,2,3]
        :type padding: [int]
        :return: list of the labels locations with the padding. Concatenated in the provided order.
        :rtype: numpy.array
        """
        # figure out what columns you want from the padded cycle
        label_in_cycle = {}
        for label in labels:
            # get label location in cycle
            label_id = np.where(np.array(self.labels.names) == label)[0]
            label_in_cycle[label] = np.where(self.labels.values == label_id)[0]
            # turn into Nx1 and 1xM for broadcasting to work
            label_in_cycle[label] = label_in_cycle[label][:, None] + np.array(padding)[None, :]
            if shift_by_padding:
                # shift by the amount of padding on the left
                label_in_cycle[label] = label_in_cycle[label] + abs(min(padding))
        selection = np.hstack([label_in_cycle[label] for label in labels])
        return selection

    def prepare_psh(self, trace, labels, padding):
        """
        Prepares signal for plotting.
        """
        # TODO : check padding format
        # prepare cycles
        trace = self.prepare_cycle(trace)
        n_cycles = trace.shape[0]
        cycle_length = trace.shape[1]

        # get the labels
        for label in labels:
            assert label in self.labels.names, f"There is no label {label} " \
                                               f"in annotation {self.annotation}"

        # pad the cycle
        trace = self.pad_cycle(trace, padding)

        selection = self.get_labels_in_cycle(labels, padding=padding, shift_by_padding=True)
        trace = select_columns(trace, selection)

        # subtract min padding to get to the unpadded values on the left
        # mod to get the unpadded (and looped) values on the right
        label_selection = np.mod(selection[0] - abs(min(padding)), cycle_length)

        return trace, label_selection

    @staticmethod
    def get_signal_split(labels, padding):
        n_padding = len(padding)

        split = []
        for i_label, _ in enumerate(labels):
            split.append(np.arange(n_padding) + n_padding * i_label)

        return split

    @staticmethod
    def get_vlines(labels, padding):
        n_padding = len(padding)

        vlines = []
        for i_label, _ in enumerate(labels):
            vlines.append(n_padding - 0.5 + n_padding * i_label)

        return vlines

    def plot_trace(self, ax, trace, mean, e, plot_individual, signal_split,
                   covariate=False, show_plot=False):
        """


        :param covariate: wether or not it's plotting covariate trace
        :type covariate: bool
        :param ax: Axis to which to add the trace
        :type ax:
        :param trace: Trace to plot, must be already reshaped to the desired shape
        :type trace:
        :param mean: Mean trace
        :type mean:
        :param e: Error bars to add
        :type e:
        :param plot_individual: Whether to plot the traces of the individual runs
        :type plot_individual:
        :param signal_split: How to split the trace along the x-axis
        :type signal_split:
        :param show_plot: Whether to show the plot
        :type show_plot:
        """
        # if you wish to not connect/disconect certain groups of signals,
        # it's indexed AFTER looping and time_points were already done:
        #  index along the x axis you will see
        if ax is None:
            ax = plt.gca()

        if covariate:
            noise_color = self.c_noise_color
            mean_color = self.c_mean_color
            edge_color = self.c_edge_color
        else:
            noise_color = self.noise_color
            mean_color = self.mean_color
            edge_color = None

        if signal_split is None:
            if plot_individual:
                ax.plot(trace.T, noise_color, alpha=0.4, linewidth=1)
            plot_errorbar(ax, mean, e, color=mean_color, edge_color=edge_color)
        else:
            for signal_group in signal_split:
                if plot_individual:
                    ax.plot(signal_group, trace[:, signal_group].T, noise_color, alpha=0.3)
                plot_errorbar(ax, mean[signal_group], e[:, signal_group],
                              x=signal_group, color=mean_color, edge_color=edge_color)

        if show_plot:
            plt.show()

    def plot_cycles(self, ax, trace_id,
                    forward_shift=None,
                    plot_individual=True,
                    ax_limits=None,
                    show_plot=False):
        """
        Places a specified trace (the signal from one cell) at the provided axis , ax.
        With the corresponding labels.
        """
        if ax is None:
            ax = plt.gca()

        # get the individual signal trace
        trace = self.get_trace(trace_id)

        # get the signals to plot in the desired shape
        trace = self.prepare_cycle(trace)
        mean, e = self.get_trace_stats(trace, error_type=self.error_type)

        # shift the signals forward
        if forward_shift is not None:
            trace = shift_signal(trace, forward_shift, axis=-1)
            mean = shift_signal(mean, forward_shift, axis=0)
            e = shift_signal(e, forward_shift, axis=-1)

        # get axis limits
        if ax_limits is None:
            xmin, xmax, ymin, ymax = get_ax_limits(trace, mean, e, plot_individual)
            ax_limits = [xmin, xmax, ymin, ymax]

        # plot the trace
        self.plot_trace(ax, trace, mean, e, plot_individual, signal_split=None)

        # create the stimuli labels in the background
        names, _, img = self.labels.plot_labels(ax,
                                                forward_shift=forward_shift,
                                                extent=ax_limits)

        # axis clean-up
        xmin, xmax, ymin, ymax = ax_limits
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        ax.set_xticks(np.arange(len(mean)))

        if show_plot:
            plt.show()

        return ax_limits

    def get_labels_for_conditions(self, conditions):
        """
        Extracts the label of the type annotation from the provided list of conditions
        :param conditions: conditions to plot, list of lists of tuples ... like this:
        [
        [(label_type,label), (label_type,label), ...],
        [(label_type,label), (label_type,label),...],
        ...
        ]
        Note: if it's a tuple of length 1, you need to write it as ("x",) not ("x") !
        :type conditions: [tup]
        :return: list of labels for each condition
        :rtype: []
        """
        condition_label = [''] * len(conditions)
        for ic, condition in enumerate(conditions):
            for label in condition:
                if label[1] in self.labels.names:
                    condition_label[ic] = label[1]
            if condition_label[ic] == "":
                raise ValueError(
                    f"condition {condition} doesn't contain a label of type {self.annotation}.")
        return condition_label

    def get_volumes_for_conditions(self, conditions):
        """
        Returns the volumes that correspond to the conditoins
        :return: list of volumes for each condition list
        :rtype:[[int]]
        """
        volumes = []
        for condition in conditions:
            volumes.append(self.experiment.choose_volumes(condition))
        return volumes

    def plot_psh(self, ax, trace_id,
                 labels,
                 padding,
                 plot_individual=True,
                 split=True,
                 ax_limits=None,
                 show_plot=False):
        """
        Places a specified trace (the signal from one cell) at the provided axis , ax.
        With the corresponding labels.
        """
        if ax is None:
            ax = plt.gca()

        # get the individual signal trace
        trace = self.get_trace(trace_id)

        # get the signals to plot in the desired shape
        trace, label_selection = self.prepare_psh(trace, labels, padding)
        mean, e = self.get_trace_stats(trace, error_type=self.error_type)

        # get axis limits
        if ax_limits is None:
            xmin, xmax, ymin, ymax = get_ax_limits(trace, mean, e, plot_individual)
            ax_limits = (xmin, xmax, ymin, ymax)
        else:
            xmin, xmax, ymin, ymax = ax_limits

        # plot the trace
        if split:
            signal_split = self.get_signal_split(labels, padding)
        else:
            signal_split = None
        self.plot_trace(ax, trace, mean, e, plot_individual, signal_split)

        # create the stimuli labels in the background
        names, _, img = self.labels.plot_labels(ax,
                                                extent=[xmin, xmax, ymin, ymax],
                                                time_points=label_selection)

        # to separate the plot regions with vertical lines
        vlines = self.get_vlines(labels, padding)
        ax.vlines(vlines, ymin, ymax, linewidth=0.8, color='black')  # , linestyle=(0, (5, 10))

        # axis clean-up
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        ax.set_xticks(np.arange(len(mean)))

        if show_plot:
            plt.show()

        return ax_limits

    def prepare_condition_trace(self, trace, conditions, padding):
        """
        Crops the signal that corresponds to the conditions
        :return: signal for each condition, concatenated
        :rtype: numpy array
        """
        labels = self.get_labels_for_conditions(conditions)
        # get the volumes to use for each condition list
        condition_vols = self.get_volumes_for_conditions(conditions)
        # for each group of volumes, prepare the signals:
        condition_signals = []
        for label, vols in zip(labels, condition_vols):
            condition_signals.append(self.crop_and_pad(trace, vols, padding))
        return np.concatenate(condition_signals, axis=1)

    def plot_covariates_psh(self, ax, trace_id,
                            conditions,
                            padding,
                            plot_individual=True,
                            split=True,
                            ax_limits=None,
                            show_plot=False):
        """
        Places a specified trace (the signal from one cell) at the provided axis , ax.
        With the corresponding labels.
        """
        if ax is None:
            ax = plt.gca()

        # get the individual signal trace
        trace = self.get_trace(trace_id)

        # get the labels according to the annotation to use
        labels = self.get_labels_for_conditions(conditions)

        # prepare stuff for the condition trace
        condition_signals = self.prepare_condition_trace(trace, conditions, padding)
        condition_mean, condition_e = self.get_trace_stats(condition_signals, error_type=self.error_type)

        # prepare stuff for the original trace
        trace, label_selection = self.prepare_psh(trace, labels, padding)
        mean, e = self.get_trace_stats(trace, error_type=self.error_type)

        # get axis limits
        if ax_limits is None:
            xmin1, xmax1, ymin1, ymax1 = get_ax_limits(trace, mean, e, plot_individual)
            xmin2, xmax2, ymin2, ymax2 = get_ax_limits(condition_signals, condition_mean, condition_e, plot_individual)
            xmin, xmax, ymin, ymax = min(xmin1, xmin2), max(xmax1, xmax2), min(ymin1, ymin2), max(ymax1, ymax2)
            ax_limits = (xmin, xmax, ymin, ymax)
        else:
            xmin, xmax, ymin, ymax = ax_limits

        # plot the trace
        if split:
            signal_split = self.get_signal_split(labels, padding)
        else:
            signal_split = None

        self.plot_trace(ax, trace, mean, e, plot_individual, signal_split)
        self.plot_trace(ax, condition_signals, condition_mean, condition_e, plot_individual, signal_split,
                        covariate=True)

        # create the stimuli labels in the background
        names, _, img = self.labels.plot_labels(ax,
                                                extent=[xmin, xmax, ymin, ymax],
                                                time_points=label_selection)

        # to separate the plot regions with vertical lines
        vlines = self.get_vlines(labels, padding)
        ax.vlines(vlines, ymin, ymax, linewidth=0.8, color='black')  # , linestyle=(0, (5, 10))

        # axis clean-up
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        ax.set_xticks(np.arange(len(mean)))

        if show_plot:
            plt.show()

        return ax_limits

    def make_psh_figure(self, traces, labels, padding,
                        main_title,
                        tittles,
                        plot_individual=False,
                        split=True,
                        figure_layout=None,
                        figsize=None,
                        gridspec_kw=None,
                        dpi=160,
                        show_plot=False):
        """
        Plots specified traces for all the traces.
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

        fig, axes = plt.subplots(n_rows, n_col,
                                 gridspec_kw=gridspec_kw,
                                 figsize=figsize, dpi=dpi)
        axes = axes.flatten()
        fig.suptitle(main_title)

        for plot_id, trace in enumerate(traces):
            ax = axes[plot_id]
            self.plot_psh(ax, trace,
                          labels,
                          padding,
                          plot_individual=plot_individual,
                          split=split)
            ax.set_title(tittles[plot_id])
            ax.set_xticklabels([])
            ax.set_ylabel('')
            ax.set_xlabel('')

        if show_plot:
            plt.show()

    def make_covariate_psh_figure(self, traces, conditions, padding,
                                  main_title,
                                  tittles,
                                  plot_individual=False,
                                  split=True,
                                  figure_layout=None,
                                  figsize=None,
                                  gridspec_kw=None,
                                  dpi=160,
                                  show_plot=False):
        """
        Plots specified traces for all the traces.
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

        fig, axes = plt.subplots(n_rows, n_col,
                                 gridspec_kw=gridspec_kw,
                                 figsize=figsize, dpi=dpi)
        axes = axes.flatten()
        fig.suptitle(main_title)

        for plot_id, trace in enumerate(traces):
            ax = axes[plot_id]
            self.plot_covariates_psh(ax, trace,
                                     conditions,
                                     padding,
                                     plot_individual=plot_individual,
                                     split=split)
            ax.set_title(tittles[plot_id])
            ax.set_xticklabels([])
            ax.set_ylabel('')
            ax.set_xlabel('')

        if show_plot:
            plt.show()

    def make_cycle_figure(self, traces,
                          main_title,
                          tittles,
                          plot_individual=True,
                          forward_shift=None,
                          figure_layout=None,
                          figsize=None,
                          gridspec_kw=None,
                          dpi=160,
                          show_plot=False):
        """
        Plots specified traces for all the traces.
        """

        if figure_layout is not None:
            n_rows = figure_layout[0]
            n_col = figure_layout[1]
        else:
            n_rows = len(traces)
            n_col = 1

        if figsize is None:
            figsize = (12, n_rows * 4)

        fig, axes = plt.subplots(n_rows, n_col,
                                 gridspec_kw=gridspec_kw,
                                 figsize=figsize, dpi=dpi)
        axes = axes.flatten()
        fig.suptitle(main_title)

        for plot_id, trace in enumerate(traces):
            ax = axes[plot_id]
            self.plot_cycles(ax, trace,
                             forward_shift=forward_shift,
                             plot_individual=plot_individual)
            ax.set_title(tittles[plot_id])
            ax.set_xticklabels([])
            ax.set_ylabel('')

        if show_plot:
            plt.show()

    def make_avg_act_scat_figure(self,
                                 labels,
                                 main_title,
                                 figure_layout,
                                 cell_numbers=None,
                                 number_cells=False, figsize=(20, 15), dpi=160,
                                 show_plot=False):
        def get_cmap(n, name='hsv'):
            '''
            from : https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
            Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
            RGB color; the keyword argument name must be a standard mpl colormap name.'''
            cmap = plt.cm.get_cmap(name, n)
            return [cmap(i_color) for i_color in np.arange(n)]

        fig, axes = plt.subplots(figure_layout[0], figure_layout[1], figsize=figsize, dpi=dpi)

        axes = axes.flatten()
        n_cells = self.traces.shape[1]
        if cell_numbers is None:
            cell_numbers = np.arange(n_cells)
        fig.suptitle(f"{main_title}\nTotal number of cells {n_cells}")

        # get activity at the stimuli
        avg_activity = []
        for label_name in labels:
            label_idx = self.experiment.choose_volumes((self.annotation, label_name))
            signal = self.traces[label_idx, :]
            avg_signal = signal.mean(axis=0)
            avg_activity.append(avg_signal)

        avg_activity = np.vstack(avg_activity)
        cmap = get_cmap(n_cells)

        label_pairs = list(combinations(np.arange(len(labels)), 2))
        # make sure all the points fit
        axis_limits = [min(0, avg_activity.min()) * 100, avg_activity.max() * 1.025 * 100]

        for i_pair, label_pair in enumerate(label_pairs):
            ax = axes[i_pair]
            # in %
            x = avg_activity[label_pair[0]] * 100
            y = avg_activity[label_pair[1]] * 100

            if number_cells:
                ax.scatter(x, y, c=cmap, alpha=0.5)
                for i, txt in enumerate(cell_numbers):
                    ax.annotate(txt, (x[i], y[i]))
            else:
                ax.scatter(x, y, c="grey", alpha=0.5)

            ax.plot(axis_limits, axis_limits, "-k", linewidth=0.5)

            ax.set_xlabel(f'Response to {labels[label_pair[0]]}, avg. dff (%)')
            ax.set_ylabel(f'Response to {labels[label_pair[1]]}, avg. dff (%)')
            ax.set_xlim(axis_limits)
            ax.set_ylim(axis_limits)
            ax.set_aspect(1.0)
        plt.tight_layout(pad=1, w_pad=1, h_pad=1)

        if show_plot:
            plt.show()
