import unittest
import tifffile as tif
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt

import vodex as vx
from numan.utils import *
from tifffile import TiffFile, imread, imwrite


class TestGaussianFilter(unittest.TestCase):
    project_dir = "D:/Code/repos/numan/notebooks/data/v1.x.x/2vs3vs5/processed"
    save_dir = "D:/Code/repos/numan/notebooks/data/v1.x.x/2vs3vs5/test_output"
    experiment = vx.Experiment.load(Path(project_dir, "experiment_raw.db"))

    def test_gaussian_filter_like_Fiji(self):
        """
        MANUAL:
        Load 3 volumes, save them to folder, blur with sigma 1 and save..
        then go to the folder, perform blur with Fiji and compare the two blurs.
        """
        three_first_volumes = self.experiment.load_volumes([0, 1, 3])
        print(type(three_first_volumes))
        print(type(three_first_volumes[0, 0, 0, 0]))

        # write image
        imwrite(f'{self.save_dir}/three_volumes.tif',
                three_first_volumes.astype(np.uint16), shape=(3, 52, 468, 500),
                metadata={'axes': 'TZYX'}, imagej=True)

        sigma = 1
        three_first_volumes = gaussian_filter(three_first_volumes, sigma)
        # write blurred image
        imwrite(f'{self.save_dir}/three_volumes_blurred.tif',
                three_first_volumes.astype(np.uint16), shape=(3, 52, 468, 500),
                metadata={'axes': 'TZYX'}, imagej=True)
