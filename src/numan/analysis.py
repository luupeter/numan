"""
Classes for Numerosity analysis.
"""
from tifffile import TiffFile, imread, imwrite
from pathlib import Path
import numpy as np
import scipy as sp
import json
import os

import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import pandas as pd
import PyPDF2

# try:
import ants
# except ImportError:
#     print("ants not installed, some registration will not work")

from .utils import *


class Spot:
    """
    This is a class for a n individual segmented spot.
    """

    def __init__(self, center, diameter, resolution=None, units='pix'):
        """
        Parameters
        ----------
        center : list or numpy array, center in zyx order
        diameter : list or numpy array or int, diameter in zyx order. If int, then it is a sphere.
        resolution : list or numpy array, resolution in xyz order.
        Defaults to [1,1,1] if not specified and only pixels are used.
        info : any extra info about the spots
        units : what units the center and diameter are using , physical 'phs', or pixels , 'pix'.
        When using physical units you must provide the resolution.
        """
        self.diameter = {}
        self.center = {}

        center = np.array(center)
        if isinstance(diameter, int):
            diameter = np.array([1, 1, 1]) * diameter
        else:
            diameter = np.array(diameter)

        if resolution is not None:
            self.resolution = np.array(resolution)
            if units == 'phs':

                self.center['phs'] = center
                self.diameter['phs'] = diameter

                self.center['pix'] = np.round(center / self.resolution)
                self.diameter['pix'] = np.round(diameter / self.resolution)
                # in case we get a 0 diameter with all the rounding
                self.diameter['pix'][self.diameter['pix'] == 0] = 1

            elif units == 'pix':
                self.center['phs'] = center * self.resolution
                self.diameter['phs'] = diameter * self.resolution

                self.center['pix'] = center
                self.diameter['pix'] = diameter
            else:
                raise AssertionError("Units can be 'pix' or 'phs' only ")

        if resolution is None:
            self.resolution = None
            if units == 'phs':
                raise AssertionError('when using physical units, you must provide the resolution')

            elif units == 'pix':

                self.center['phs'] = None
                self.diameter['phs'] = None

                self.center['pix'] = center
                self.diameter['pix'] = diameter
            else:
                raise AssertionError("Units can be 'pix' or 'phs' only ")

    def __str__(self):
        return f"Spot at {self.center['phs']} phs, {self.center['pix']} pix\n" \
               f"diameter {self.diameter['phs']} phs, {self.diameter['pix']} pix\n" \
               f"resolution {self.resolution}. Everything in ZYX order."

    def __repr__(self):
        return self.__str__()

    def mask_at_zero(self, diameter=None, units='pix'):
        """
        diameter : list or numpy array or int, diameter in zyx order. If int, then it is a sphere.
                    If not None, diameter will be used to create a mask.
                    If None, then spots diameter will be used and units will be ignored.
        units: what units is the diameter in : 'pix' or 'phs'
        """

        if diameter is None:
            d = self.diameter['pix']
        else:
            if isinstance(diameter, int):
                diameter = np.array([1, 1, 1]) * diameter
            else:
                diameter = np.array(diameter)
            if units == 'phs':
                diameter = np.round(diameter / self.resolution)
            d = diameter

        r = d / 2  # radius
        # make sure diameter is odd: round to the next odd number
        d = ((d // 2) * 2 + 1).astype(int)
        # center at zero
        c = (d // 2).astype(int)
        z = np.arange(d[0]) - c[0]
        y = np.arange(d[1]) - c[1]
        x = np.arange(d[2]) - c[2]

        # find what pixels are inside the ellipsoid
        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        inside_ellipsoid = np.sqrt((zz / r[0]) ** 2 + (yy / r[1]) ** 2 + (xx / r[2]) ** 2) <= 1

        # get a 3D mask
        mask = np.zeros((d[0], d[1], d[2]))
        mask[inside_ellipsoid] = 1

        # get list of nonzero elements , centered at zero
        i, j, k = np.where(mask > 0)
        centered_idx = np.c_[i, j, k] - c

        return mask, centered_idx

    def create_mask(self, volumes, diameter=None, units='pix'):
        """
        Creates a binary mask that can be applied to the volume.
        volumes : image as a sequence of volumes in shape TZYX or one volume in ZYX
        diameter : list or numpy array or int, diameter in zyx order. If int, then it is a sphere.
                    If not None, diameter will be used to create a mask.
                    If None, then spots diameter will be used and units will be ignored.
        units: what units is the diameter in : 'pix' or 'phs'
        """
        # create mask for a single volume
        if len(volumes.shape) == 4:
            t, zmax, ymax, xmax = volumes.shape
        elif len(volumes.shape) == 3:
            zmax, ymax, xmax = volumes.shape
        else:
            raise AssertionError("volumes should be 4D tzyx or 3D zyx")
        mask = np.zeros((zmax, ymax, xmax))

        # get list of nonzero elements , centered at zero
        _, idx = self.mask_at_zero(diameter=diameter, units=units)

        # shift to center
        shift = self.center['pix']
        # because of the rounding when going from phs to pix, some extreme spots can be outside the image
        # so make sure they are inside
        border = np.array([zmax - 1, ymax - 1, xmax - 1])
        shift = np.min(np.c_[shift, border], axis=1)

        # get the indices
        idx = idx + shift.astype(int)

        # remove the ones outside the boundary
        idx = idx[np.all(idx >= 0, axis=1), :]
        is_inside = np.logical_and(idx[:, 0] < zmax, np.logical_and(idx[:, 1] < ymax, idx[:, 2] < xmax))
        idx = idx[is_inside, :]

        mask[idx[:, 0], idx[:, 1], idx[:, 2]] = 1

        if np.sum(mask) == 0:
            warnings.warn(f"The whole spot centered at {self.center['pix']} pix ( {self.center['phs']} phs), with "
                          f"diameter {self.diameter['pix']} pix ({self.diameter['phs']} phs) is "
                          f"outside of the volume.")

        return mask, idx

    @staticmethod
    def plot_mask(mask, figsize=(8, 6), dpi=160):
        n_slices = mask.shape[0]
        if n_slices > 1:
            f, ax = plt.subplots(1, mask.shape[0], figsize=figsize, dpi=dpi)
            for islice, iax in enumerate(ax):
                iax.imshow(mask[islice, :, :], vmin=0, vmax=1)
                iax.set_axis_off()
                iax.set_title(f"z {islice}")
                iax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        else:
            plt.imshow(mask[0, :, :], vmin=0, vmax=1)


class Signals:
    def __init__(self, traces, traces_type="raw", fps=1):
        """
        traces : matrix TxN ( time x number of signals )
        traces_type : a 3 letter code for what kind of signals are these : "raw", "dff", "zsc" ( z  - score )
        """
        self.traces = np.array(traces)
        self.T, self.N = self.traces.shape

        self.traces_type = traces_type
        self.fps = fps

    def change_fs(self, fps):
        self.fps = fps

    def as_dff(self, window_size):
        """
        Returns dff of the design_matrix, the beginning of the proper measurement and the end
        ( such measurement that used the whole window for definition of baseline )
        """
        assert self.traces_type == "raw", f"Can't apply dff: " \
                                          f"the signals have already been processed, these are {self.traces_type} signals"
        dff, start, end = get_dff(self.traces, window_size)

        return Signals(dff, traces_type="dff")

    def hp_filter(self, cutoff):
        """
        Returns high-pass filtered signals from the design_matrix
        """
        assert self.traces_type == "raw", f"Can't apply dff: " \
                                          f"the signals have already been processed, these are {self.traces_type} signals"

        def butter_highpass(cutoff, fps, order=5):
            # calculate the numerator and denominator coefficient vectors of the filter
            nyq = 0.5 * fps
            normal_cutoff = cutoff / nyq
            b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
            return b, a

        def butter_highpass_filter(data, cutoff, fps, order=5):
            # return the filtered signal
            b, a = butter_highpass(cutoff, fps, order=order)
            y = signal.filtfilt(b, a, data, axis=0)
            return y

        filtered_traces = butter_highpass_filter(self.traces, cutoff, self.fps)

        return Signals(filtered_traces, traces_type="hp filtered", fps=self.fps)

    def as_zscore(self, cutoff, window_size):
        """
        Returns z-score of the design_matrix
        Uses high-pass filter for z-score calculation
        """
        assert self.traces_type == "raw", f"Can't apply dff: " \
                                          f"the signals have already been processed, these are {self.traces_type} signals"
        # get baseline : as median
        percentile = 50
        baseline, start, end = get_baseline(self.traces, window_size, percentile)
        # get standart deviation from high-pass filtred data
        filtered_signals = self.hp_filter(cutoff)
        filtered_sd = np.std(filtered_signals.traces, axis=0)
        # calcualte z-score
        zscore_signals = (self.traces - baseline) / filtered_sd

        return Signals(zscore_signals, traces_type="zscore", dff=self.dff)

    @classmethod
    def from_spots(cls, spots, volumes=None, experiment=None, batch_size=None, movie=None,
                   verbose=False, traces_type="raw"):
        """
        Extracts signals from volumes for each spot in spots.
        spots : list[Spot]
        volumes : volume idx (list[int]) to load and extract signals from, or "all"
        movie : volume sequence (tzyx) to extract signals from.
        """

        assert (movie is not None) or (volumes is not None and experiment is not None), \
            "Provide a movie or volume IDs and an experiment "

        if volumes == "all":
            # only use full volumes
            volumes = experiment.list_volumes()
            volumes = volumes[volumes >= 0]

        def fill_signal(matrix, movie, t_start, t_end):
            for isp, spot in enumerate(spots):
                _, idx = spot.create_mask(movie)
                # gets the pixels in the spot mask
                avg_signal = np.mean(movie[:, idx[:, 0], idx[:, 1], idx[:, 2]], axis=1)
                matrix[t_start:t_end, isp] = avg_signal
            return matrix

        N = len(spots)

        design_matrix = None
        # if volumes are given explicitly as a movie
        if movie is not None:
            T = movie.shape[0]
            design_matrix = np.zeros((T, N))
            design_matrix = fill_signal(design_matrix, movie, 0, T)

        # if you need to load volumes from disk
        elif volumes is not None and experiment is not None:
            T = len(volumes)
            design_matrix = np.zeros((T, N))
            # split volumes into batches of certain size
            if batch_size is not None:
                btcs = np.array_split(volumes, np.ceil(T / batch_size))
            else:
                btcs = volumes

            t_start = 0
            for batch in btcs:
                t_end = t_start + len(batch)
                if verbose:
                    print(f"Batch {batch}, t_start {t_start}, t_end {t_end}")
                # load a batch from disk
                movie = experiment.load_volumes(batch, verbose=True)
                # extract signals for all the spots from the loaded chunk
                design_matrix = fill_signal(design_matrix, movie, t_start, t_end)

                t_start = t_end

        return cls(design_matrix, traces_type=traces_type)

    def get_looped(self, trace, experiment, time_points=None, cycles=None, error_type="prc"):
        """
        Returns signals looped per cycle
        time_points: time points of the cycle. If you only need certain time-points from the cycle ( in volumes )
        if you need to stack timepoints for each cycle, for example: [[0,1,2],[7,8,9]], pass the time_points as 2D list.
        """

        n_cycles = experiment.full_cycles
        # cycle length in volumes
        cycle_length = experiment.cycle.full_length / experiment.volume_manager.fpv
        assert cycle_length.is_integer(), "Need more info to determine the cycle..." \
                                          "by which I mean that plot_looped function needs more code :), " \
                                          "Sorry. Hint : use time_points "
        assert self.T == (n_cycles * cycle_length), "Need more info to determine the cycle..." \
                                                    "by which I mean that plot_looped function needs more code :), " \
                                                    "Sorry. Hint : use time_points "
        cycled = self.traces[:, trace].reshape((n_cycles, int(cycle_length)))

        # grab only the desired cycles
        if cycles is not None:
            cycles = np.array(cycles)
            cycled = cycled[cycles, :]

        # crop out the necessary cycle part,
        # note: you can reorder the cycle as well ( to add points from the beginning  to the end of the cycle)
        # you can also overlay some cycle time intervals, useful for psh
        if time_points is not None:
            time_points = np.array(time_points)
            time_shape = time_points.shape
            assert len(time_shape) < 3, "time_shape should be 1D or 2D"
            cycled = cycled[:, time_points.flatten()]
            # if you want to overlay cycle intervals:
            if len(time_shape) == 2:
                cycled = cycled.reshape(-1, time_shape[1])

        mean = np.mean(cycled, axis=0)

        if error_type == "prc":
            # error bars : 5 to 95 th percentile around the median
            e = np.r_[np.expand_dims(mean - np.percentile(cycled, 5, axis=0), axis=0),
            np.expand_dims(np.percentile(cycled, 95, axis=0) - mean, axis=0)]
        elif error_type == "sem":
            # error bars : sem around hte mean
            sem = np.std(cycled, axis=0, ddof=1) / np.sqrt(cycled.shape[0])
            e = np.r_[np.expand_dims(sem, axis=0),
            np.expand_dims(sem, axis=0)]
        else:
            e = None

        return cycled, mean, e

    def to_csv(self, filename):
        """
        saves signals to csv.
        """
        df = pd.DataFrame(self.traces, columns=[f"cell {icell}" for icell in np.arange(self.N)])
        df.to_csv(filename)


class Spots:

    def __init__(self, spots, groups=None, signals=None):
        """
        Parameters
        ----------
        spots : list[Spot]
        groups : dictionary of type {"group_name" : list[bool]} that assigns cells to various groups
        signals : Signals for the corresponding spots
        """

        self.spots = spots
        self.num_spots = len(self.spots)
        self.signals = signals
        # TODO : have signals as dict with " raw" , " dff" etc
        self.groups = None
        if groups is not None:
            self.add_groups(groups)

    def __str__(self):

        groups = "No"
        if self.groups is not None:
            groups = self.groups.keys()

        signals = "not loaded"
        if self.signals is not None:
            signals = "loaded"

        return f"{self.num_spots} spots\n " \
               f"{groups} groups\n" \
               f"Signals {signals}"

    def __repr__(self):
        return self.__str__()

    def add_groups(self, groups, rewrite=False):
        """
        groups: dict with boolean arrays to say which cells belong.
        """
        if not rewrite and self.groups is not None:
            for group in groups:
                assert group not in self.groups, f"The group {group} already exists," \
                                                 f" use rewrite to overwrite it," \
                                                 f" or use another group name"
        if self.groups is None:
            self.groups = {}
        for group in groups:
            # TODO : add protection from rewriting
            self.groups[group] = np.array(groups[group])

    def list_groups(self):
        if self.groups is not None:
            for key in self.groups.keys():
                print(key)

    def _get_centers(self, units='phs'):
        """
        returns a list of lists
        """
        centers = []
        for spot in self.spots:
            centers.append(spot.center[units].tolist())
        return centers

    def _get_diameters(self, units='phs'):
        diameters = []
        for spot in self.spots:
            diameters.append(spot.diameter[units].tolist())
        return diameters

    def _get_resolutions(self):
        resolutions = []
        for spot in self.spots:
            resolutions.append(spot.resolution.tolist())
        return resolutions

    def to_json(self, filename, units='phs'):
        """
        Transform Spots object into json format and save as a file.
        """

        j_dict = {
            "resolutions": self._get_resolutions(),
            "diameters": self._get_diameters(units=units),
            "centers": self._get_centers(units=units),
            "units": units,
            "groups": None}

        if self.groups is not None:
            j_dict["groups"] = {}
            for group in self.groups:
                j_dict["groups"][group] = np.array(self.groups[group], dtype=bool).tolist()

        if self.signals is not None:
            j_dict["signals"] = {"traces": self.signals.traces.tolist(),
                                 "traces_type": self.signals.traces_type}
        else:
            j_dict["signals"] = None

        j = json.dumps(j_dict)

        with open(filename, 'w') as json_file:
            json_file.write(j)

    @classmethod
    def from_json(cls, filename):
        """
        Load Spots object from json file.
        """
        # create an object for the class to return
        with open(filename) as json_file:
            j = json.load(json_file)

        spots = []
        units = j["units"]
        for center, diameter, resolution in zip(j["centers"], j["diameters"], j["resolutions"]):
            spots.append(Spot(center, diameter, resolution=resolution, units=units))

        groups = j["groups"]

        if j["signals"] is not None:
            signals = Signals(j["signals"]["traces"], traces_type=j["signals"]["traces_type"])
        else:
            signals = None

        return cls(spots, groups=groups, signals=signals)

    @classmethod
    def from_imaris(cls, points_file, diameter_file, resolution=None, units='phs'):
        """
        Load Spots using imaris csvs:
        Creates list of spots from the output of Imaris segmentation.

        position_file: *_Position.csv , file containing segmented centers
        diam_file: *_Diameter.csv , file containing diameters
        """
        p_df = pd.read_csv(points_file, skiprows=[0, 1, 2])
        d_df = pd.read_csv(diameter_file, skiprows=[0, 1, 2])
        centers = p_df[['Position Z', 'Position Y', 'Position X']].to_numpy()
        diams = d_df[['Diameter Z', 'Diameter Y', 'Diameter X']].to_numpy()
        spots = []
        for center, diam in zip(centers, diams):
            spots.append(Spot(center, diam, resolution=resolution, units=units))

        return cls(spots, groups=None, signals=None)

    def get_signals(self, volumes=None, experiment=None, batch_size=None, movie=None, traces_type="raw", reload=False):
        assert reload or self.signals is None, "Spots already have signals loaded, reload? "

        self.signals = Signals.from_spots(self.spots, volumes=volumes, experiment=experiment,
                                          batch_size=batch_size, movie=movie, traces_type=traces_type)

    def get_group_mask(self, group, mask_shape, diameter=None, units='pix'):
        """
        Create a 3D volume that only shows the spots that belong to the particular group
        group : list[bool], the length of spots
        mask_shape : the shape of the 3D volume of the mask
        diameter : list or numpy array or int, diameter in zyx order. If int, then it is a sphere.
                    If not None, diameter will be used to create a mask.
                    If None, then spots diameter will be used and units will be ignored.
        units: what units is the diameter in : 'pix' or 'phs'
        """
        mask = np.zeros(mask_shape)
        for ispot in tqdm(np.where(group)[0]):
            imask, _ = self.spots[ispot].create_mask(mask, diameter=diameter, units=units)
            mask = mask + imask
        return mask

    def get_group_signals(self, group):
        """
        Returns signals for a particular group only
        """
        traces = self.signals.traces[:, group]
        return Signals(traces, traces_type=self.signals.traces_type)

    def get_group_idx(self, group):
        """
        Returns indices for a particular group only
        """
        idx = np.arange(self.num_spots)
        idx = idx[group]
        return idx

    def get_group_centers(self, group, units='pix'):
        """
        Returns centers for a particular group only
        units: the units of the center to return
        """
        centers = []
        for ispot, spot in enumerate(self.spots):
            if group[ispot]:
                centers.append(spot.center[units])

        return np.array(centers)

    def get_group_info(self, group_list, group=None):
        """
        Returns a string with the titles of the groups from group_list, where each spot is a member.
        group = the T/F list ... If group is not None: only returns it for the group.
        """
        group_info = []
        for ispot, spot in enumerate(self.spots):
            groups = ""
            for group_name in group_list:
                if self.groups[group_name][ispot]:
                    groups = groups + group_name + "; "
            group_info.append(groups)
        group_info = np.array(group_info)
        if group is not None:
            group_info = group_info[group]
        return group_info


class SignalAnalyzer:

    def __init__(self, signals):
        self.signals = signals

    @staticmethod
    def bootstrap_distribution_of_difference_of_means(data, split, nbootstrap=10000):
        """
        Creates a bootstrap distribution of the absolute difference of the means for the two groups

        data : list of data to split into the two groups and calculate the difference of the mean
        split : [n_group1,n_group2]
        nbootstrap : number of bootstrap repetitions
        """
        N = data.shape[0]
        if np.sum(split) != N:
            raise AssertionError("sum of split should equal the number of data points")
        delta = []
        for ib in np.arange(nbootstrap):
            group1 = np.random.choice(N, size=split[0], replace=True)
            group2 = np.arange(N)[~group1]
            delta.append(np.mean(data[group1]) - np.mean(data[group2]))
        return delta

    @staticmethod
    def get_p_of_difference_of_means(data1, data2, nbootstrap=10000):

        split = [len(data1), len(data2)]
        merged_data = np.r_[data1, data2]

        delta = np.abs(np.mean(data1) - np.mean(data2))
        # get distribution of the absolute diff of the means
        delta_dist = SignalAnalyzer.bootstrap_distribution_of_difference_of_means(merged_data, split=split,
                                                                                  nbootstrap=nbootstrap)
        delta_dist = np.abs(delta_dist)

        n_samples = len(delta_dist)
        n_extreme = np.sum(delta_dist >= delta)
        p = n_extreme / n_samples
        return p

    def get_p_list_of_difference_of_means(self, group1, group2, nbootstrap=10000):
        """
        group1 : data points ( time points, in volumes ) to assign to group 1
        group2 : data points ( time points, in volumes ) to assign to group 2
        """
        data1 = self.signals.traces[group1, :]
        data2 = self.signals.traces[group2, :]
        T1, nspots = data1.shape
        T2 = data1.shape[0]
        # print(f" Data points in group 1 : {T1},in group 1 : {T2}.\nNumber of spots : {nspots}.")

        p = []
        for ispot in tqdm(np.arange(nspots)):
            p.append(self.get_p_of_difference_of_means(data1[:, ispot], data2[:, ispot], nbootstrap=nbootstrap))
        return p


class Volumes:
    """
    Collection of volumes and functions that perform operations on them.
    """

    def __init__(self, volumes, resolution=[1, 1, 1]):
        self.volumes = volumes
        self.resolution = resolution
        self.shape = volumes.shape

    def average_volumes(self):
        return Volumes(np.mean(self.volumes, axis=0))

    def voxelise(self):
        pass

    def get_dff(self, window_size):
        """
        Returns dff of the volumes (assuming 0 dimension is time),
        the beginning of the proper measurement and the end
        ( such measurement that used the whole window for definition of baseline )
        """
        dff, start, end = get_dff(self.volumes, window_size)
        return dff, start, end


class Preprocess:
    """
    Collection of methods to perform preprocessing of the raw data in experiment.
    """

    def __init__(self, experiment):
        self.experiment = experiment

    def remove_offset(self, save_dir, batch_size, volumes = None, offset_img=None, offset_file=None, verbose=False):
        """
        Denoises the raw movie by subtracting the offset image.
        The offset image can be prepared using coscmos package.
        Will only process full volumes.
        """
        assert offset_img is not None or offset_file is not None, \
            "Either offset_img or offset_file should be provided"
        assert not (offset_img is not None and offset_file is not None), \
            "Only one of offset_img or offset_file should be provided"
        if offset_img is None:
            offset_img = imread(offset_file)

        # Split the volumes into chunks of the defined size that will be loaded into RAM at once
        # if volumes are not provided, will use all the full volumes
        # if volumes are provided, will only process the requested volumes
        chunks = self.experiment.batch_volumes(batch_size, volumes=volumes, full_only=True)
        
        if verbose:
            tot_volumes = 0
            n_requested_volumes = self.experiment.n_full_volumes
            if volumes is not None:
                n_requested_volumes = len(volumes)

        for ichunk, chunk in enumerate(chunks):
            data = self.experiment.load_volumes(chunk, verbose=False)
            data = data - offset_img
            nt, z, y, x = data.shape

            # make sure the output is positive:
            min_value = data.min()
            if min_value < 0:
                warnings.warn(f"Some image values belov zero after subtracting the offset image. "
                              f"Will be set to zero.")
                data[data < 0] = 0

            # write image
            imwrite(f'{save_dir}/removed_offset_movie_{ichunk:04d}.tif',
                    data.astype(np.int16), shape=(nt, z, y, x),
                    metadata={'axes': 'TZYX'}, imagej=True)

            if verbose:
                tot_volumes = tot_volumes + nt
                print(f"written volumes : {tot_volumes}, out of {n_requested_volumes} full volumes")

    def drift_correct_naive(self, save_dir, batch_size, spacing_xyz, template = 0, 
                            registration_type = "Rigid", volumes = None, verbose=False, 
                            ants_kwargs = {}):
        """
        Performs drift correction using ANTs library for registration. 
        template can be an int (a frame to register to), or a 3D numpy array.
        If template is not provided, the first volume will be used as a template.
        if volumes is not None, then only the volumes in the list will be registered.
        For more info, see ANTs documentation: https://antspy.readthedocs.io/en/latest/registration.html

        Args:
            save_dir (str): directory to save the drift corrected movie.
            batch_size (int): number of volumes to load at once when processing batches. Each batch is saved as a separate file.
            spacing_xyz (list): spacing in xyz order
            template (int or 3D numpy array): template to register to
            registration_type (str): type of registration to perform, see ANTs documentation for available options.
            volumes (list): list of volumes to register. If None, all the full volumes will be registered.
            verbose (bool): whether to print the volumes that have been processed so far on the screen.
            ants_kwargs (dict): parameters for the registration. If empty, default parameters will be used. See ANTs documentation for available options.
                mask and moving_mask can be provided as paths to the masks, they will be converted to ants images.
                Note: (see https://stackoverflow.com/questions/334655/passing-a-dictionary-to-a-function-as-keyword-parameters)
                 1. ants.registration can have parameters that are not included in the dictionary, just specify what you want, the rest will be default. 
                 2. You can not override an ants.registration function parameter that is already in the dictionary, 
                 so don't try to include keywords fixed, moving, type_of_transform, verbose and outprefix in the dictionary.
                 3. The dictionary can not have values that aren't in the ants.registration function.
        """
        def _register(moving_image, fixed_image, spacing, registration_type, outprefix = None):

            # if fixed image in not ants image, convert it
            if not isinstance(fixed_image, ants.core.ants_image.ANTsImage):
                print("Converting fixed image to ants image")
                fixed_image = ants.from_numpy(np.transpose(fixed_image.astype(float), axes=[2,1,0]), 
                                    spacing=spacing) # spacing in xyz order
                
            moving_image = ants.from_numpy(np.transpose(moving_image.astype(float), axes=[2,1,0]), 
                                spacing=spacing)
            
            # run the registration and save verbose to file
            rr = ants.registration(
                                    fixed=fixed_image,
                                    moving=moving_image,
                                    type_of_transform=registration_type, 
                                    verbose  = False,
                                    outprefix = outprefix,
                                    **ants_kwargs
                                )
            return rr['warpedmovout'].numpy().astype(np.int16).transpose((2,1,0))
        
        def _register_batch(batch, template, spacing, registration_type, outprefix = None):
            corrected = []
            for volume in batch:
                # load the volume
                volume_img = self.experiment.load_volumes([volume], verbose=False)[0]
                # register
                volume_img = _register(volume_img, template, spacing, registration_type, 
                                       outprefix = outprefix)
                corrected.append(volume_img)
            corrected = np.array(corrected)
            return corrected

        # prepare list of volumes to register
        n_full_volumes = self.experiment.n_full_volumes
        if volumes is not None:
            volumes = np.array(volumes)
            assert volumes.max() < n_full_volumes,\
                  f"Some volumes are outside of the experiment with {n_full_volumes} full volumes" 
            assert volumes.min() >= 0, "volumes must be positive integers"
            n_reguested_volumes = len(volumes)
        else:
            volumes = np.arange(n_full_volumes)
            n_reguested_volumes = n_full_volumes

        # if template is not an array, get template volume
        if isinstance(template, int):
            assert template < n_full_volumes, f"Template frame {template} is outside of the experiment with {n_full_volumes} full volumes"
            assert template >= 0, "Template frame must be a positive integer"
            template = self.experiment.load_volumes([template], verbose=False)[0]
        # turn template into ants image
        template = ants.from_numpy(np.transpose(template.astype(float), axes=[2,1,0]), 
                                    spacing=spacing_xyz) # spacing in xyz order
        
        # if mask and moving_mask are in ants_kwargs, convert them to ants images
        # and if they are string or pthlib.Path, load them as numpy arrays first
        if 'mask' in ants_kwargs:
            if isinstance(ants_kwargs['mask'], str) or isinstance(ants_kwargs['mask'], Path):
                ants_kwargs['mask'] = ants.from_numpy(np.transpose(imread(ants_kwargs['mask']).astype(float), axes=[2,1,0]), 
                                    spacing=spacing_xyz)
            elif isinstance(ants_kwargs['mask'], np.ndarray):
                ants_kwargs['mask'] = ants.from_numpy(np.transpose(ants_kwargs['mask'].astype(float), axes=[2,1,0]), 
                                    spacing=spacing_xyz)

        if 'moving_mask' in ants_kwargs:
            if isinstance(ants_kwargs['moving_mask'], str) or isinstance(ants_kwargs['moving_mask'], Path):
                ants_kwargs['moving_mask'] = ants.from_numpy(np.transpose(imread(ants_kwargs['moving_mask']).astype(float), axes=[2,1,0]), 
                                    spacing=spacing_xyz)
            elif isinstance(ants_kwargs['moving_mask'], np.ndarray):
                ants_kwargs['moving_mask'] = ants.from_numpy(np.transpose(ants_kwargs['moving_mask'].astype(float), axes=[2,1,0]), 
                                    spacing=spacing_xyz)
        
        # load in batches and process volume by volume
        chunks = self.experiment.batch_volumes(batch_size, volumes= volumes, full_only=True)
        # create a directory for the transforms
        os.makedirs(f'{save_dir}/transforms', exist_ok=True)

        tot_volumes = 0
        for ichunk, chunk in enumerate(chunks):
            # register the chunk
            outprefix = f'{save_dir}/transforms/volume_{chunk[0]:04d}'
            corrected = _register_batch(chunk, template, spacing_xyz, registration_type,
                                        outprefix = outprefix)
            # save 
            nt, z, y, x = corrected.shape
            imwrite(f'{save_dir}/drift_corrected_movie_{ichunk:04d}.tif',
                    corrected.astype(np.int16), shape=(nt, z, y, x),
                    metadata={'axes': 'TZYX'}, imagej=True)
            if verbose:
                tot_volumes = tot_volumes + nt
                print(f"written volumes : {tot_volumes}, out of {n_reguested_volumes} requested volumes")


    def batch_dff_sliding_window(self, save_dir, batch_size, window_size, blur_sigma=None, verbose=False):
        """
        Creates 3D dff movie from raw 3D movie using sliding window. Will only use full_volumes,
        so the number of frames in the resulting movie can be smaller than in the original.

        :param blur_sigma: If not None, will apply gaussian blur in 3D with sigma = blur_sigma.
                         Can be int - then the same sigma in all 3 directions is applied,
                         or list [sz,sy,sx] for different sigma in ZYX.
        :type blur_sigma: Union(int, list)
        :param save_dir: directory into which to save the dff movie in chunks
        :type save_dir: str
        :param batch_size: number of volumes to load at once
        :type batch_size: int
        :param window_size: the size of the sliding window for the baseline estimation in volumes
        :type window_size: int
        :param verbose: whether to print the volumes that have been processed so far on the screen.
        :type verbose: bool
        """
        # TODO : make the size & digit estimation
        # TODO : write resolution into metadata

        # will only use full volumes
        volume_list = self.experiment.list_volumes()
        volume_list = volume_list[volume_list >= 0]
        n_volumes = len(volume_list)

        # will multiply dff image by this value for better visualisation later
        SCALE = 1000

        # break volumes into chunks that will be downloaded at once
        overlap = window_size - 1
        # there are more chunks than needed, this is okay:
        # the cycle will end after the first chunk that contains the end of the experiment
        chunks = [volume_list[i:i + batch_size] for i in range(0, len(volume_list), batch_size - overlap)]

        for ich, chunk in enumerate(tqdm(chunks, disable=verbose)):

            data = self.experiment.load_volumes(chunk, verbose=False)
            if blur_sigma is not None:
                data = gaussian_filter(data, blur_sigma)
            dff_img, start_tp, end_tp = get_dff(data, window_size)
            t, z, y, x = dff_img.shape

            # special case of the last chunk --> need to output the tail as well, and then break
            if chunk[-1] == (n_volumes - 1):
                end_tp = t
            # special case of the first chunk --> need to output the head as well
            if ich == 0:
                start_tp = 0

            # make sure the output fits into the int16 range:
            min_value = dff_img[start_tp:end_tp, :, :, :].min()
            max_value = dff_img[start_tp:end_tp, :, :, :].max()
            if max_value * SCALE > 32767 or min_value * SCALE < -32768:
                warnings.warn(f"Scaled DFF values outside the int16 range for scale = {SCALE},"
                              f" min: {min_value}, max: {max_value}")
            # write image
            imwrite(f'{save_dir}/dff_movie_{ich:04d}.tif',
                    (dff_img[start_tp:end_tp, :, :, :] * SCALE).astype(np.int16), shape=(end_tp - start_tp, z, y, x),
                    metadata={'axes': 'TZYX'}, imagej=True)

            if verbose:
                print(f"written frames : {chunk[start_tp]} - {chunk[end_tp - 1]}, out of {n_volumes}")
            # exit cycle the first chunk you saw the end of the experiment
            if chunk[-1] == (n_volumes - 1):
                break

    def calculate_dff(self, save_dir, batch_size, step_size, baseline_volumes,
                      resolution_xyz = [1, 1, 1], blur_sigma=None, 
                      verbose=False, save_baseline=False):
        """
        Creates 3D dff movie from raw 3D movie by processing movie in steps. Will only use full_volumes,
        so the number of frames in the resulting movie can be smaller than in the original.

        Args:
            blur_sigma (Union(int, list)): If not None, will apply gaussian blur in 3D with sigma = blur_sigma.
                            Can be int - then the same sigma in all 3 directions is applied,
                            or list [sz,sy,sx] for different sigma in ZYX.
            save_dir (str): directory into which to save the dff movie in chunks
            batch_size (int): number of volumes to load at once, must be multiple of step_size
            step_size (int): the size of the sliding window for the baseline estimation in volumes
            baseline_volumes (List[int]): volumes inside each step to use for baseline estimation
            resolution_xyz (List[int]): resolution in xyz order in um
            verbose (bool): whether to print the volumes that have been processed so far on the screen.
        """
        # TODO : make the size & digit estimation
        # TODO : write resolution into metadata

        assert batch_size % step_size == 0, "batch_size must be multiple of step_size"

        if save_baseline:
            #create directory for baseline
            os.makedirs(f'{save_dir}/baseline', exist_ok=True)

        # will only use full volumes
        volume_list = self.experiment.list_volumes()
        volume_list = volume_list[volume_list >= 0]
        n_volumes = len(volume_list)

        # will multiply dff image by this value for better visualisation later
        SCALE = 1000

        #create chunks
        chunks = self.experiment.batch_volumes(batch_size, volumes=volume_list, full_only=True)

        for ich, chunk in enumerate(tqdm(chunks, disable=verbose)):

            data = self.experiment.load_volumes(chunk, verbose=False)
            # add 1 to all pixels to aviod division by zero
            data = data + 1

            if blur_sigma is not None:
                data = gaussian_filter(data, blur_sigma)
            dff_img, baseline_img = get_dff_in_steps(data, step_size, baseline_volumes)
            t, z, y, x = dff_img.shape

            # special case of the last chunk --> need to output the tail as well, and then break
            if chunk[-1] == (n_volumes - 1):
                end_tp = t
            # special case of the first chunk --> need to output the head as well
            if ich == 0:
                start_tp = 0

            # make sure the output fits into the int16 range:
            min_value = dff_img.min()
            max_value = dff_img.max()
            if max_value * SCALE > 32767 or min_value * SCALE < -32768:
                warnings.warn(f"Scaled DFF values outside the int16 range for scale = {SCALE},"
                              f" min: {min_value}, max: {max_value}")
            # write dff image
            imwrite(f'{save_dir}/dff_movie_{ich:04d}.tif', (dff_img * SCALE).astype(np.int16), shape=(t, z, y, x),
                    resolution=(1./resolution_xyz[0], 1./resolution_xyz[1]), metadata={'spacing': resolution_xyz[2], 'unit': 'um','axes': 'TZYX'}, 
                    imagej=True)
            if save_baseline:
                # print(f"Baseline shape{baseline_img.shape}")
                # write baseline image
                imwrite(f'{save_dir}/baseline/baseline_movie_{ich:04d}.tif', baseline_img.astype(np.int16), shape=(t, z, y, x),
                        resolution=(1./resolution_xyz[0], 1./resolution_xyz[1]), metadata={'spacing': resolution_xyz[2], 'unit': 'um','axes': 'TZYX'}, 
                        imagej=True)

            if verbose:
                print(f"written frames : {chunk[0]} - {chunk[- 1]}, out of {n_volumes}")
            # exit cycle the first chunk you saw the end of the experiment
            if chunk[-1] == (n_volumes - 1):
                break


if __name__ == "__main__":
    pass
