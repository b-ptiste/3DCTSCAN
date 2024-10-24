import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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

    _, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(list_filter):
        ax.errorbar(
            x, list_means[i], yerr=list_stds[i], label=method, fmt="-o", capsize=5
        )

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
        "#FFD700",
        "#40E0D0",
        "#FF4500",
        "#ADFF2F",
        "#FF6347",
        "#FF69B4",
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


def plot_confusion_matrix(filter_name: list[str], map_res_val: dict, with_agg: bool):
    """
    This function is used to plot the confusion matrix
    :param filter_name: list of filter names
    :param map_res_val: dict
    """

    name_conf = "confusion_matrix" if not with_agg else "confusion_matrix_agg"

    # Create subplots
    figure, axes = plt.subplots(1, len(filter_name), figsize=(15, 5))

    # Loop through methods and axes together
    for ax, name in zip(axes, filter_name):
        cm = map_res_val[name][name_conf] * 100 / np.sum(map_res_val[name][name_conf])

        # Plot the confusion matrix on the respective axis
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=True,
            vmin=0.0,
            vmax=100.0,
            ax=ax,
        )

        # Add labels and title
        ax.set_title(f"Confusion Matrix for {name}")
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")

    plt.tight_layout()
    plt.show()


def plot_threshold_relative_error(
    th_thres: list[float],
    pred_thres: list[float],
    pred_thres_agg: list[float],
    filter_name: list[str],
):
    """
    This function is used to plot the threshold relative error.
    :param th_thres: list of threshold
    :param pred_thres: list of predicted threshold
    :param pred_thres_agg: list of predicted threshold aggregated
    :param filter_name: list of filter names
    """
    plt.figure(figsize=(10, 6))
    x = list(range(len(filter_name)))

    # Plot the threshold and predicted values
    plt.plot(x, th_thres, "bo-", label="threshold", color="blue")
    plt.plot(x, pred_thres, "ro-", label="pred_threshold", color="orange")
    plt.plot(x, pred_thres_agg, "ro-", label="pred_threshold_agg", color="red")
    # Add vertical lines and annotate with relative errors
    for i in range(len(x)):
        plt.vlines(
            x[i], pred_thres[i], th_thres[i][0], colors="gray", linestyles="dashed"
        )

        # Calculate relative error
        rel_error = abs((th_thres[i][0] - pred_thres[i]) / th_thres[i][0]) * 100

        plt.text(
            x[i],
            (th_thres[i][0] + pred_thres[i]) / 2,
            f"{rel_error:.1f}%",
            ha="center",
            va="bottom",
            color="black",
        )

    plt.xticks(x, filter_name)
    plt.ylabel("Values")
    plt.legend()

    # Show plot
    plt.show()
