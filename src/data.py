import os

import torch
import numpy as np
import SimpleITK as sitk
from skimage import measure

from .utils import save_sample


class Dataset(torch.utils.data.Dataset):
    """
    This class is used to load the dataset of CT scans.
    """

    def __init__(self, nb_sample, prepro):
        self.nb_sample = nb_sample
        self.prepro = prepro

    def __getitem__(self, index):
        return self.load_samples(index)

    def __len__(self):
        return self.nb_sample

    def get_names_from_index(self, index):
        return (
            f"lungs_0{index}.nii.gz",
            f"vol_0{index}.nii.gz",
            f"vessels_0{index}.nii.gz",
        )

    def load_sample(self, filepath):
        image = sitk.ReadImage(filepath)
        array = sitk.GetArrayFromImage(image)
        return array

    def load_samples(self, index):
        index += 1
        assert index > 0 and index <= 6, "Index must be in range 1 to 6"
        names = self.get_names_from_index(index)
        samples = {}
        for name in names:
            path = name if not self.prepro else f"prepro_{name}"
            if not os.path.isfile(path):
                sample = self.load_sample(name)
                if name.split("_")[0] == "lungs":
                    # get minimal box
                    sample = np.where(sample > 0, 1, 0)
                    bbox = measure.regionprops(measure.label(sample))[0].bbox

                sample = sample[bbox[0] : bbox[3], bbox[1] : bbox[4], bbox[2] : bbox[5]]

                if name.split("_")[0] == "vol":
                    # min max scaling
                    sample = (sample - sample.min()) / (sample.max() - sample.min())
                save_sample(sample, path)
            else:
                sample = self.load_sample(path)
            samples[name.split("_")[0]] = sample
        return samples
