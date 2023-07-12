import tifffile as tif
import numpy as np
import json
import os

import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import pandas as pd

import PyPDF2
from PyPDF2 import PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import white, black

from pathlib import Path

from .plots import *
from .analysis import Spots


def merge_pdfs(pdfs, filename):
    """
    Turns a bunch of separate figures (pdfs) into one pdf.
    """
    mergeFile = PyPDF2.PdfMerger()
    for pdf in pdfs:
        mergeFile.append(PyPDF2.PdfReader(pdf, 'rb'))
        os.remove(pdf)
    mergeFile.write(filename)


def place_cb(can, x, y, name):
    form = can.acroForm
    can.setFont("Courier", 12)
    can.drawCentredString(x + 20, y + 20, name)
    form.checkbox(name=name,
                  # tooltip = f"Field {name}",
                  x=x + 10,
                  y=y - 4,
                  # buttonStyle = 'check',
                  borderColor=black,
                  fillColor=white,
                  textColor=black,
                  forceBorder=True
                  )
    return can


def create_checkbox_pdf(pngs, cells_idx, btchs, filename):
    """
    Turns a bunch of separate figures (pngs) into one pdf with checkboxes.
    """
    # create an empty pdf
    can = canvas.Canvas(filename, pagesize=letter)
    # for each page:
    for ibtch, btch in enumerate(btchs):
        # refresh checkbox locations to be at the top,
        # numbers fitted to match the plots in the shape (5,1)
        X, Y, H = 10, 600, 122
        # add image to pdf and delete png file
        can.drawImage(pngs[ibtch], 0, 0, width=650,
                      preserveAspectRatio=True, mask='auto')
        # add checkboxes
        for cell_name in cells_idx[btch]:
            can = place_cb(can, x=X, y=Y, name=cell_name)
            Y = Y - H
        # finish page
        can.showPage()
        os.remove(pngs[ibtch])
    can.save()


class Reports:
    """
    For now it is simply a class of wrappers to make reports specifically for the 2vs3vs5 experiment.
    I hope it will become more clean and general as time goes on.
    """

    def __init__(self, project_folder, experiment):
        self.project = project_folder
        self.experiment = experiment

    @staticmethod
    def prepare_spot_info(spots, group_tag, n_traces, sort_by_sig=True,
                          groups_to_specify=("sig2v3", "sig2v5", "sig3v5", "sig2vB", "sig3vB", "sig5vB")):
        # some info on the cells to put into the title
        # cells idx in the original set of cells as they are in the spots
        cells_idx = spots.get_group_idx(spots.groups[group_tag])
        # cells coordinates
        cells_zyx = spots.get_group_centers(spots.groups[group_tag]).astype(np.int32)
        # list of groups that the cell belongs to ( from the list provided )
        cells_group = spots.get_group_info(groups_to_specify,
                                           group=spots.groups[group_tag])
        cells_group = np.array([group_name.replace("sig", "") for group_name in cells_group])
        # since signals are already cropped for the group, it's just np.arrange
        signal_idx = np.arange(n_traces)

        if sort_by_sig:
            # sort everything so that the cells with the most amount of significant stuff appear first
            sorted_zip = sort_by_len0(zip(cells_group, cells_idx, cells_zyx, signal_idx))
            cells_group = np.array([el[0] for el in sorted_zip])
            cells_idx = np.array([el[1] for el in sorted_zip])
            cells_zyx = np.array([el[2] for el in sorted_zip])
            signal_idx = np.array([el[3] for el in sorted_zip])

        return cells_group, cells_idx, cells_zyx, signal_idx

    def make_signal_reports(self,
                            spot_tag, group_tag,
                            annotation_type,
                            dff_param,
                            labels=None,
                            plot_type="cycle",
                            plot_type_tag='',
                            forward_shift=0,
                            plot_individual=False,
                            groups_to_specify=("sig2v3", "sig2v5", "sig3v5", "sig2vB", "sig3vB", "sig5vB"),
                            tmp_folder=None,
                            pdf_filename=None,
                            checkbox=False):
        """
        Generates a pdf with the specified type of plots.

        :param checkbox:
        :type checkbox:
        :param groups_to_specify:
        :type groups_to_specify:
        :param labels:
        :type labels:
        :param annotation_type:
        :type annotation_type:
        :param dff_param: parameters for dff calculation (see Signals.as_dff).
            For example {"method":"sliding", "window":15} or {"method":"step", "step_size":9, "baseline_volumes": [0,1,2]}
        :type dff_param: dict
        :param spot_tag: what set of spots to use. Chooses the spots*.json based on this.
        :type spot_tag: str
        :param group_tag: what spots group to use.
        :type group_tag: str
        :param plot_type: what plot to output ["cycle","psh_0","psh_b"]
        :type plot_type: str
        :param plot_type_tag: just for the pdf naming : this is to be able to distinguish
                    the pdfs with the same plot type, but errors are different or raw traces on/off or front_to_tail...
        :type plot_type_tag: str
        :param forward_shift: forward_shift will shift the cycle by the set number of voxels
                    so when set to 3, there are 3 blank volumes at the begining and at the end ...
                    if set to 0, will have 6 leading blanks and will end right after the 5 dots (black bar)
        :type forward_shift: int
        :param time_points: only show certain timepoints from the signal, for example : only 2 dots.
                    IF time_points is 2d array, will overlap traces along axis = 1.
        :type time_points: numpy.array
        :param vlines: draw vertical lines, locations (in volumes) where to draw vertical lines
        :type vlines: list[int]
        :param signal_split: how to break the lines in the plot, this is relevant to the displayed x axis
        :type signal_split: numpy.array
        :param error_type: what error type to use ( "sem" for SEM or "prc" for 5th - 95th percentile )
        :type error_type: str
        :param noise_color: the color of the individual traces (if shown)
        :type noise_color: valid color definition
        :param plot_individual: wheather to plot the individual traces
        :type plot_individual: bool
        :param tmp_folder: will store batch images in this folder before merging pdf,
                    will be stored with reports if left None.
        :type tmp_folder: Union(str, Path)
        :param pdf_filename: the name of the pdf file to save, will be generated automatically if left None
        :type pdf_filename: str

        """
        if plot_type == "cycle":
            assert dff_param["method"] == "sliding", "Cycle plots only work with sliding dff"
        if plot_type == "psh_0" or plot_type == "psh_b":
            assert labels is not None, "Labels must not be None for psh-type plots"

        # fill out the defaults
        if tmp_folder is None:
            if checkbox:
                # where to temporary store images while the code is running
                tmp_folder = f"{self.project}/spots/reports/groupped/signals/"
            else:
                # where to temporary store images while the code is running
                tmp_folder = f"{self.project}/spots/reports/all_significant/signals/"

        if pdf_filename is None:
            if checkbox:
                # filename to save pdf with all the significant traces and checkboxes
                pdf_filename = f"{self.project}/spots/reports/groupped/signals/" \
                               f"CHOOSE_{plot_type}{plot_type_tag}_from_{spot_tag}_group_{group_tag}.pdf"
            else:
                # filename to save pdf with all the significant traces
                pdf_filename = f"{self.project}/spots/reports/all_significant/signals/" \
                               f"{plot_type}{plot_type_tag}_from_{spot_tag}_group_{group_tag}.pdf"

        spots = Spots.from_json(f"{self.project}/spots/signals/spots_{spot_tag}.json")

        # initialise the signal plotter with DFF signal
        print(f"Using the following parameters for signal DFF: {dff_param}")
        signals = spots.get_group_signals(spots.groups[group_tag]).as_dff(**dff_param)
        s_plotter = SignalPlotter(signals, self.experiment, annotation_type)

        # prepare title info
        cells_group, cells_idx, cells_zyx, signal_idx = self.prepare_spot_info(spots, group_tag,
                                                                               s_plotter.n_traces,
                                                                               sort_by_sig=True,
                                                                               groups_to_specify=groups_to_specify)
        main_title = f"DFF signals, tscore image {spot_tag}, group {group_tag}"

        # choose traces per page
        if plot_type == "psh_0":
            if checkbox:
                raise AssertionError("Can not build checkboxes for plot type 'psh_0' ")
            tpp = 10  # traces per page
        else:
            tpp = 5

        # prepare the batches per page
        cells = np.arange(s_plotter.n_traces)
        btchs = [cells[s: s + tpp] for s in np.arange(np.ceil(s_plotter.n_traces / tpp).astype(int)) * tpp]

        plot_files = []

        for ibtch, btch in enumerate(btchs):

            if plot_type == "cycle":
                # titles for the current batch
                titles = [f"Cell {idx}, {group} XYZ : {zyx[2]},{zyx[1]},{zyx[0]} (voxel) "
                          for idx, group, zyx in zip(cells_idx[btch], cells_group[btch], cells_zyx[btch])]

                s_plotter.make_cycle_figure(signal_idx[btch],
                                            main_title,
                                            titles,
                                            forward_shift=forward_shift,
                                            # what grid to use to show the points
                                            figure_layout=[5, 1],
                                            # figure parameters
                                            figsize=(10, 12),
                                            dpi=60,
                                            # whether to plot the individual traces
                                            plot_individual=plot_individual)

            if plot_type == "psh_0":
                padding = [0]
                # titles for the current batch
                titles = [f"Cell {idx}, {group} \nXYZ : {zyx[2]},{zyx[1]},{zyx[0]} (voxel) "
                          for idx, group, zyx in zip(cells_idx[btch], cells_group[btch], cells_zyx[btch])]
                s_plotter.make_psh_figure(signal_idx[btch],
                                          labels, padding,
                                          main_title,
                                          titles,
                                          # what grid to use to show the points
                                          figure_layout=[5, 2],
                                          # figure parameters
                                          figsize=(10, 12),
                                          dpi=60,
                                          gridspec_kw={'hspace': 0.4, 'wspace': 0.3},
                                          # whether to plot the individual traces
                                          plot_individual=plot_individual, split=False)

            if plot_type == "psh_b":
                padding = [-3, -2, -1, 0, 1, 2, 3, 4, 5]
                # titles for the current batch
                titles = [f"Cell {idx}, {group} XYZ : {zyx[2]},{zyx[1]},{zyx[0]} (voxel) "
                          for idx, group, zyx in zip(cells_idx[btch], cells_group[btch], cells_zyx[btch])]
                s_plotter.make_psh_figure(signal_idx[btch],
                                          labels, padding,
                                          main_title,
                                          titles,
                                          # what grid to use to show the points
                                          figure_layout=[5, 1],
                                          # figure parameters
                                          figsize=(10, 12),
                                          dpi=60,
                                          # whether to plot the individual traces
                                          plot_individual=plot_individual)
            plt.xlabel('Volume in cycle')
            if checkbox:
                filename = f'{tmp_folder}signals_batch{ibtch}.png'
            else:
                filename = f'{tmp_folder}signals_batch{ibtch}.pdf'
            plt.savefig(filename)
            plt.close()
            plot_files.append(filename)

        if checkbox:
            cells_idx_txt = np.array([str(idx) for idx in cells_idx])
            create_checkbox_pdf(plot_files, cells_idx_txt, btchs, pdf_filename)
        else:
            merge_pdfs(plot_files, pdf_filename)

    def make_avg_intensity_reports(self,
                                   spot_tag, group_tag,
                                   annotation_type,
                                   dff_param,
                                   labels,
                                   number_cells=False,
                                   plot_type_tag='',
                                   pdf_filename=None):

        if pdf_filename is None:
            # filename to save pdf with all the significant traces
            pdf_filename = f"{self.project}/spots/reports/all_significant/signals/" \
                           f"avg_intensity_scatterplot{plot_type_tag}_from_{spot_tag}_group_{group_tag}.pdf"

        spots = Spots.from_json(f"{self.project}/spots/signals/spots_{spot_tag}.json")
        cells_idx = spots.get_group_idx(spots.groups[group_tag])

        # initialise the signal plotter with DFF signal
        print(f"Using the following parameters for signal DFF: {dff_param}")
        signals = spots.get_group_signals(spots.groups[group_tag]).as_dff(**dff_param)
        s_plotter = SignalPlotter(signals, self.experiment, annotation_type)

        # prepare title info
        main_title = f"Average DFF per stimuli, pairwise comparison."

        # figure out figure layout
        if len(labels) == 3:
            figure_layout = [1, 3]
        elif len(labels) == 4:
            figure_layout = [2, 3]
        elif len(labels) == 5:
            figure_layout = [2, 5]
        else:
            raise Exception(f"Don't know how to plot when the number of labels is {len(labels)}")

        s_plotter.make_avg_act_scat_figure(labels, main_title,
                                           cell_numbers=cells_idx,
                                           # what grid to use to show the points
                                           figure_layout=figure_layout,
                                           # figure parameters
                                           figsize=(12, 10),
                                           dpi=160,
                                           number_cells=number_cells)

        plt.savefig(pdf_filename)
        plt.close()

    def make_covariate_reports(self,
                               spot_tag, group_tag,
                               annotation_type,
                               dff_param,
                               conditions=None,
                               plot_type="cycle",
                               plot_type_tag='',
                               forward_shift=0,
                               plot_individual=False,
                               groups_to_specify=("sig2v3", "sig2v5", "sig3v5", "sig2vB", "sig3vB", "sig5vB"),
                               tmp_folder=None,
                               pdf_filename=None):
        """
        Generates a pdf with the specified type of plots.

        :param checkbox:
        :type checkbox:
        :param groups_to_specify:
        :type groups_to_specify:
        :param conditions:
        :type conditions:
        :param annotation_type:
        :type annotation_type:
        :param spot_tag: what set of spots to use. Chooses the spots*.json based on this.
        :type spot_tag: str
        :param group_tag: what spots group to use.
        :type group_tag: str
        :param plot_type: what plot to output ["cycle","psh_0","psh_b"]
        :type plot_type: str
        :param plot_type_tag: just for the pdf naming : this is to be able to distinguish
                    the pdfs with the same plot type, but errors are different or raw traces on/off or front_to_tail...
        :type plot_type_tag: str
        :param forward_shift: forward_shift will shift the cycle by the set number of voxels
                    so when set to 3, there are 3 blank volumes at the begining and at the end ...
                    if set to 0, will have 6 leading blanks and will end right after the 5 dots (black bar)
        :type forward_shift: int
        :param time_points: only show certain timepoints from the signal, for example : only 2 dots.
                    IF time_points is 2d array, will overlap traces along axis = 1.
        :type time_points: numpy.array
        :param vlines: draw vertical lines, locations (in volumes) where to draw vertical lines
        :type vlines: list[int]
        :param signal_split: how to break the lines in the plot, this is relevant to the displayed x axis
        :type signal_split: numpy.array
        :param error_type: what error type to use ( "sem" for SEM or "prc" for 5th - 95th percentile )
        :type error_type: str
        :param noise_color: the color of the individual traces (if shown)
        :type noise_color: valid color definition
        :param plot_individual: wheather to plot the individual traces
        :type plot_individual: bool
        :param tmp_folder: will store batch images in this folder before merging pdf,
                    will be stored with reports if left None.
        :type tmp_folder: Union(str, Path)
        :param pdf_filename: the name of the pdf file to save, will be generated automatically if left None
        :type pdf_filename: str

        """
        if plot_type == "cycle":
            assert dff_param["method"] == "sliding", "Cycle plots only work with sliding dff"

        assert conditions is not None, "Conditions must not be None for psh-type plots"

        # fill out the defaults
        if tmp_folder is None:
            # where to temporary store images while the code is running
            tmp_folder = f"{self.project}/spots/reports/covariates/signals/"

        if pdf_filename is None:
            # filename to save pdf with all the significant traces
            pdf_filename = f"{self.project}/spots/reports/covariates/signals/" \
                           f"{plot_type}{plot_type_tag}_from_{spot_tag}_group_{group_tag}.pdf"

        spots = Spots.from_json(f"{self.project}/spots/signals/spots_{spot_tag}.json")

        # initialise the signal plotter with DFF signal
        print(f"Using the following parameters for signal DFF: {dff_param}")
        signals = spots.get_group_signals(spots.groups[group_tag]).as_dff(**dff_param)

        # choose traces per page
        if plot_type == "psh_0":
            tpp = 10  # traces per page
            s_plotter = SignalPlotter(signals, self.experiment, annotation_type,
                                      c_mean_color='w', c_noise_color='-m', c_edge_color='w')
        else:
            tpp = 5
            s_plotter = SignalPlotter(signals, self.experiment, annotation_type,
                                      c_mean_color='k', c_noise_color='-m', c_edge_color='w')

        # prepare title info
        cells_group, cells_idx, cells_zyx, signal_idx = self.prepare_spot_info(spots, group_tag,
                                                                               s_plotter.n_traces,
                                                                               sort_by_sig=True,
                                                                               groups_to_specify=groups_to_specify)
        main_title = f"DFF signals, tscore image {spot_tag}, group {group_tag}"

        # prepare the batches per page
        cells = np.arange(s_plotter.n_traces)
        btchs = [cells[s: s + tpp] for s in np.arange(np.ceil(s_plotter.n_traces / tpp).astype(int)) * tpp]

        plot_files = []

        for ibtch, btch in enumerate(btchs):

            if plot_type == "psh_0":
                padding = [0]
                # titles for the current batch
                titles = [f"Cell {idx}, {group} \nXYZ : {zyx[2]},{zyx[1]},{zyx[0]} (voxel) "
                          for idx, group, zyx in zip(cells_idx[btch], cells_group[btch], cells_zyx[btch])]
                s_plotter.make_covariate_psh_figure(signal_idx[btch],
                                                    conditions, padding,
                                                    main_title,
                                                    titles,
                                                    # what grid to use to show the points
                                                    figure_layout=[5, 2],
                                                    # figure parameters
                                                    figsize=(10, 12),
                                                    dpi=60,
                                                    gridspec_kw={'hspace': 0.4, 'wspace': 0.3},
                                                    # whether to plot the individual traces
                                                    plot_individual=plot_individual, split=False)

            if plot_type == "psh_b":
                padding = [-3, -2, -1, 0, 1, 2, 3, 4, 5]
                # titles for the current batch
                titles = [f"Cell {idx}, {group} XYZ : {zyx[2]},{zyx[1]},{zyx[0]} (voxel) "
                          for idx, group, zyx in zip(cells_idx[btch], cells_group[btch], cells_zyx[btch])]
                s_plotter.make_covariate_psh_figure(signal_idx[btch],
                                                    conditions, padding,
                                                    main_title,
                                                    titles,
                                                    # what grid to use to show the points
                                                    figure_layout=[5, 1],
                                                    # figure parameters
                                                    figsize=(10, 12),
                                                    dpi=60,
                                                    # whether to plot the individual traces
                                                    plot_individual=plot_individual)
            plt.xlabel('Volume in cycle')
            filename = f'{tmp_folder}signals_batch{ibtch}.pdf'
            plt.savefig(filename)
            plt.close()
            plot_files.append(filename)

        merge_pdfs(plot_files, pdf_filename)


class CellMasks:
    def __init__(self, project, experiment):
        self.project = project
        self.experiment = experiment
        self.filenames, self.new_groups, self.spot_tags = self.get_new_groups()

    def get_new_groups(self, verbose=True):
        group_filenames = list(Path(self.project.main_folder,
                                    'processed/spots/reports/groupped/signals/').glob('[*'))
        new_groups = [str(name).split('[')[1].split(']')[0] for name in group_filenames]
        spot_tags = [str(name).split('from_')[1].split('_group')[0] for name in group_filenames]
        if verbose:
            print(pd.DataFrame({"new group": new_groups, "from spots": spot_tags}))
        return group_filenames, new_groups, spot_tags

    @staticmethod
    def get_checked_boxes(filename):
        """get checked boxes from pdf"""

        f = open(filename, "rb")
        group_pdf = PyPDF2.PdfReader(f)
        fields = group_pdf.get_fields()
        spots_in_group = np.array([spot for spot in fields if
                                   fields[spot]['/V'] == '/Yes']).astype(int)
        f.close()
        return spots_in_group

    def add_groups_from_pdf(self, rewrite=False):
        spots = Spots.from_json(f"spots/signals/spots_{self.spot_tags[0]}.json")
        current_spots = self.spot_tags[0]
        for filename, group_tag, spot_tag in zip(self.filenames, self.new_groups, self.spot_tags):
            if spot_tag != current_spots:
                spots.to_json(f"spots/signals/spots_{current_spots}.json")
                # prepare the new group
                current_spots = spot_tag
                spots = Spots.from_json(f"spots/signals/spots_{spot_tag}.json")

            # get checked boxes
            spots_in_group = self.get_checked_boxes(filename)
            is_in_group = np.zeros((spots.num_spots,)).astype(bool)
            is_in_group[spots_in_group] = True

            # add the new groups to spots
            spots.add_groups({group_tag: is_in_group}, rewrite=rewrite)
            print(f"Added group {group_tag} to spots_{spot_tag}.json")
        # save the last chunk
        spots.to_json(f"spots/signals/spots_{current_spots}.json")

    def write_masks(self, resolution):
        # get the size of one volume
        T, Z, Y, X = self.experiment.load_volumes([0]).shape
        print(f"Image shape : {T}, {Z}, {Y}, {X}")

        # resolution in ZYX order in um
        image_dir = Path(self.project.main_folder, "processed", "spots", "reports", "groupped", "images")

        spots = Spots.from_json(f"spots/signals/spots_{self.spot_tags[0]}.json")
        current_spots = self.spot_tags[0]
        for filename, group_tag, spot_tag in zip(self.filenames, self.new_groups, self.spot_tags):
            if spot_tag != current_spots:
                current_spots = spot_tag
                spots = Spots.from_json(f"spots/signals/spots_{spot_tag}.json")

            mask = spots.get_group_mask(spots.groups[group_tag], (Z, Y, X))
            tif.imwrite(f'{image_dir}/mask_from_{spot_tag}_group_{group_tag}.tif',
                        mask.astype(np.uint16), shape=(Z, Y, X),
                        metadata={'spacing': resolution[0], 'unit': 'um', 'axes': 'ZYX'},
                        resolution=(1 / resolution[1], 1 / resolution[2]), imagej=True)
            filename.rename(Path(filename.parent, f"done_{filename.name}"))
            print(f"Created mask for spots from {spot_tag} tscore image, for group {group_tag}")
