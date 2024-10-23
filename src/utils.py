import numpy as np


def get_stats(map_res_train: dict):
    """
    This function is used to get the statistics of the model.
    :param map_res: dict
    :return: dict
    """
    map_res_stats = {}
    for k1, v1 in map_res_train.items():
        map_res_stats[k1] = {}
        for k2, v2 in v1.items():
            map_res_stats[k1][k2] = {}
            for k3, v3 in v2.items():
                if k3 in ["roc_auc", "best_threshold"]:
                    map_res_stats[k1][k2][f"{k3}_mean"] = np.mean(v3)
                    map_res_stats[k1][k2][f"{k3}_std"] = np.std(v3)
                map_res_stats[k1][k2]["roc_auc_aug"] = v2["roc_auc_agg"][0]
                map_res_stats[k1][k2]["best_threshold_aug"] = v2["best_threshold_agg"][
                    0
                ]

    return map_res_stats