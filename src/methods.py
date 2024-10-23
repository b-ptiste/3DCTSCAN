import os

import SimpleITK as sitk
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.filters import meijering, sato, frangi, hessian
from skimage import measure
from sklearn.metrics import roc_curve, auc, confusion_matrix
from collections import defaultdict
from tqdm import tqdm


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


def save_sample(array: np.ndarray, filepath: str):
    """
    This function is used to save a numpy array as a .nii.gz file.
    :param array: numpy array to save
    :param filepath: path to save the file
    :return: None
    """
    # Convert numpy array back
    image = sitk.GetImageFromArray(array)

    # Save the image as a .nii.gz file
    sitk.WriteImage(image, filepath)


def compute_roc_curve(vesselness: np.ndarray, vessels_ref: np.ndarray):
    """
    This function is used to compute the ROC curve and the best threshold.
    :param vesselness: numpy array of vesselness values
    :param vessels_ref: numpy array of reference vessel values
    :return: fpr, tpr, roc_auc, best_threshold
    """
    vessels_ref_flat = vessels_ref.flatten()
    vesselness_flat = vesselness.flatten()

    fpr, tpr, thresholds = roc_curve(vessels_ref_flat, vesselness_flat)
    roc_auc = auc(fpr, tpr)
    best_threshold = thresholds[np.argmax(tpr - fpr)]

    return fpr, tpr, roc_auc, best_threshold


def get_map_struct(names_filter: list[str], configs: dict):
    """
    This function construct a storage structure
    :param names_filter: list of  filter names
    :param config: list of filter configurations
    :return: dict
    """
    return {
        name: {
            f'{config["sigmas"].start}_{config["sigmas"].stop}_{config["sigmas"].step}': defaultdict(
                list
            )
            for config in configs
        }
        for name in names_filter
    }


def get_filter(name: str):
    """
    This function is used to get the filter function.
    :param name: name of the filter
    :return: filter function
    """
    assert name in [
        "frangi",
        "sato",
        "meijering",
        "hessian",
    ], f"filter must be in {name}"
    filter = {
        "frangi": frangi,
        "sato": sato,
        "meijering": meijering,
        "hessian": hessian,
    }[name]

    return filter


def train(
    dataset: torch.utils.data.Dataset,
    configs: dict,
    indexes: list[int],
    filter_name: list[str],
):
    """
    This function is used to train the model.
    :param dataset: dataset
    :param configs: list of filter configurations
    :param indexes: list of indexes
    :param filter_name: list of filter names
    :return: dict
    """
    map_res = get_map_struct(filter_name, configs)
    print("...Strarts")
    for index in tqdm(indexes):
        ## > Get Data
        samples = dataset[index]
        vol_ct = samples["vol"]
        lungs_mask = samples["lungs"] > 0
        vessels_ref = samples["vessels"]

        ## > Compute the filters
        for config in configs:
            for name in filter_name:
                vessels_pred = get_filter(name)(vol_ct, **config)

                ## > Compute ROC curve and threshold
                fpr, tpr, roc_auc, best_threshold = compute_roc_curve(
                    vessels_pred[lungs_mask], vessels_ref[lungs_mask]
                )
                save_sample(
                    vessels_pred,
                    f"/content/output_0{index+1}_train_bf_th_{name}.nii.gz",
                )
                save_sample(
                    (vessels_pred > best_threshold) * 1.0,
                    f"/content/output_0{index+1}_train_th_{name}.nii.gz",
                )
                save_sample(
                    (vessels_pred > best_threshold) * lungs_mask * 1.0,
                    f"/content/output_0{index+1}_train_{name}.nii.gz",
                )
                name_expe = f'{config["sigmas"].start}_{config["sigmas"].stop}_{config["sigmas"].step}'
                map_res[name][name_expe]["roc_auc"].append(roc_auc)
                map_res[name][name_expe]["best_threshold"].append(best_threshold)
                map_res[name][name_expe]["fpr"].append(fpr)
                map_res[name][name_expe]["tpr"].append(tpr)
                map_res[name][name_expe]["pred_stack"] = map_res[name][name_expe][
                    "pred_stack"
                ] + list(vessels_pred[lungs_mask])
                map_res[name][name_expe]["ref_stack"] = map_res[name][name_expe][
                    "ref_stack"
                ] + list(vessels_ref[lungs_mask])

    for config in configs:
        for name in filter_name:
            name_expe = f'{config["sigmas"].start}_{config["sigmas"].stop}_{config["sigmas"].step}'
            fpr, tpr, roc_auc, best_threshold = compute_roc_curve(
                np.array(map_res[name][name_expe]["pred_stack"]),
                np.array(map_res[name][name_expe]["ref_stack"]),
            )
            map_res[name][name_expe]["roc_auc_agg"].append(roc_auc)
            map_res[name][name_expe]["best_threshold_agg"].append(best_threshold)
            map_res[name][name_expe]["fpr_agg"].append(fpr)
            map_res[name][name_expe]["tpr_agg"].append(tpr)

    return map_res


def val(
    dataset: torch.utils.data.Dataset,
    configs: dict,
    indexes: list[int],
    filter_name: list[str],
):
    """
    This function is used to validate the model.
    :param dataset: dataset
    :param configs: list of filter configurations
    :param indexes: list of indexes
    :param filter_name: list of filter names
    :return: dict
    """
    map_res = {
        name: {
            "confusion_matrix": np.array([[0, 0], [0, 0]]),
            "roc_auc": [],
            "best_threshold": [],
            "fpr": [],
            "tpr": [],
            "pred_stack": [],
            "ref_stack": [],
            "roc_auc_agg": [],
            "best_threshold_agg": [],
            "fpr_agg": [],
            "tpr_agg": [],
        }
        for name in filter_name
    }

    for index in tqdm(indexes):
        ## > Get Data
        samples = dataset[index]
        vol_ct = samples["vol"]
        lungs_mask = samples["lungs"] > 0
        vessels_ref = samples["vessels"]

        ## > Compute the filters
        for name in filter_name:
            vessels_pred = get_filter(name)(
                vol_ct,
                **{
                    k: v
                    for k, v in configs[name].items()
                    if k in ["sigmas", "black_ridges"]
                },
            )
            ## > Prediction
            fpr, tpr, roc_auc, best_threshold = compute_roc_curve(
                vessels_pred[lungs_mask], vessels_ref[lungs_mask]
            )
            map_res[name]["roc_auc"].append(roc_auc)
            map_res[name]["best_threshold"].append(best_threshold)
            map_res[name]["fpr"].append(fpr)
            map_res[name]["tpr"].append(tpr)
            map_res[name]["pred_stack"] = map_res[name]["pred_stack"] + list(
                vessels_pred[lungs_mask]
            )
            map_res[name]["ref_stack"] = map_res[name]["ref_stack"] + list(
                vessels_ref[lungs_mask]
            )

            vessels_pred = (vessels_pred > configs[name]["threshold"]) * 1.0

            ## > Compute confusion metrics
            map_res[name]["confusion_matrix"] = map_res[name][
                "confusion_matrix"
            ] + confusion_matrix(
                vessels_ref[lungs_mask].flatten(), vessels_pred[lungs_mask].flatten()
            )

            save_sample(
                vessels_pred * lungs_mask, f"/content/output_0{index+1}_{name}.nii.gz"
            )
        for name in filter_name:
            fpr, tpr, roc_auc, best_threshold = compute_roc_curve(
                np.array(map_res[name]["pred_stack"]),
                np.array(map_res[name]["ref_stack"]),
            )
            map_res[name]["roc_auc_agg"].append(roc_auc)
            map_res[name]["best_threshold_agg"].append(best_threshold)
            map_res[name]["fpr_agg"].append(fpr)
            map_res[name]["tpr_agg"].append(tpr)
    return map_res
