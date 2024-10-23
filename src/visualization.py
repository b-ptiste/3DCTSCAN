import matplotlib.pyplot as plt
import numpy as np


def plot_ROC(
    list_filter: list[str],
    list_fpr: list[float],
    list_tpr: list[float],
    list_roc: list[float],
    list_best_threshold: list[float],
):
    """
    This function is used to plot the ROC curve.
    :param list_filter: list of filter names
    :param list_fpr: list of fpr
    :param list_tpr: list of tpr
    :param list_roc: list of roc
    :param list_best_threshold: list of best threshold
    """
    plt.figure()
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    for fpr, tpr, roc, best_threshold in zip(
        list_fpr, list_tpr, list_roc, list_best_threshold
    ):
        plt.plot(
            fpr,
            tpr,
            color="blue",
            lw=2,
            label=f"{list_filter} (AUC = {roc}), Best threshold = {best_threshold}",
        )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Courbe ROC")
    plt.legend(loc="lower right")
    plt.show()


def plot_res(
    list_filter: list[str],
    list_means: list[float],
    list_stds: list[float],
    _type: list[str],
    params: list[str],
):
    """
    This function is used to plot the results.
    :param list_filter: list of filter names
    :param list_means: list of means
    :param list_stds: list of stds
    :param _type: type of the plot
    :param params: list of parameters
    """
    x = np.arange(len(params))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(list_filter):
        ax.errorbar(
            x, list_means[i], yerr=list_stds[i], label=method, fmt="-o", capsize=5
        )

    # Customize plot
    ax.set_xticks(x)
    ax.set_xticklabels(params)
    ax.set_xlabel("Parameters")
    ax.set_ylabel(f"{_type}")
    ax.set_title(f"Comparison of {_type} Mean and Standard Deviation by Method")
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_roc_curves_train(data: dict, with_only_agg: bool = True):
    """
    This function is used to plot the ROC curves.
    :param data: dict
    :param with_only_agg: bool
    """
    plt.figure(figsize=(10, 7))
    j = 0
    colors = [
        "#FF5733",
        "#33FF57",
        "#3357FF",
        "#FF33A1",
        "#A133FF",
        "#FF9933",
        "#33FFF1",
        "#8D33FF",
        "#FF5733",
        "#FF33F4",
        "#33D4FF",
        "#FF33B8",
    ]
    for filter_name, experiments in data.items():
        for expe_name, metrics in experiments.items():
            fpr = metrics["fpr_agg"][0]
            tpr = metrics["tpr_agg"][0]
            roc_auc = metrics["roc_auc_agg"][0]
            label = f"{filter_name}_{expe_name}_ROC curve_{np.round(roc_auc, 3)}"
            plt.plot(fpr, tpr, lw=3, label=label, color=colors[j])

            if not with_only_agg:
                for i in range(len(metrics["roc_auc"])):
                    fpr = metrics["fpr"][i]
                    tpr = metrics["tpr"][i]
                    roc_auc = metrics["roc_auc"][i]
                    plt.plot(fpr, tpr, lw=2, color=colors[j], linestyle="--")
            j += 1

    # Plot settings
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Various Filters and Experiments")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def plot_roc_curves_val(data: dict, with_only_agg: bool = True):
    """
    This function is used to plot the ROC curves.
    :param data: dict
    :param with_only_agg: bool
    """
    plt.figure(figsize=(10, 7))

    for filter_name, metrics in data.items():
        print(filter_name)
        if filter_name == "frangi":
            color = "blue"
        elif filter_name == "sato":
            color = "red"
        elif filter_name == "meijering":
            color = "green"
        elif filter_name == "hessian":
            color = "orange"

        fpr = metrics["fpr_agg"][0]
        tpr = metrics["tpr_agg"][0]
        roc_auc = metrics["roc_auc_agg"][0]
        label = f"{filter_name}_ROC_curve {roc_auc} "
        plt.plot(fpr, tpr, lw=3, label=label, color=color)

        if not with_only_agg:
            for i in range(len(metrics["roc_auc"])):
                fpr = metrics["fpr"][i]
                tpr = metrics["tpr"][i]
                roc_auc = metrics["roc_auc"][i]
                plt.plot(fpr, tpr, lw=2, color=color, linestyle="--")

    # Plot settings
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Various Filters and Experiments")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
