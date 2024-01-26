"""Module to plot best possible model for every (train, test) pair of
datasets."""
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

DATA_ROOT = "results/"
NROWS, NCOLS = 2, 3
BAR_WIDTH = 0.2
CMAP_COLOR_SCHEME = "Spectral"


def fetch_all_results():
    files_list = [DATA_ROOT + x for x in os.listdir(DATA_ROOT) if x.endswith("json")]
    return files_list


def max_in_json(data):
    mx = 0
    mx_label = "NOT_FOUND"
    for prep, peaks in data.items():
        for peak, acc in peaks.items():
            if acc > mx:
                mx = acc
                mx_label = f"{prep}_{peak}"
    return mx, mx_label


def short_name(model_name):
    return model_name.replace("xgboost", "xgb").replace("naive_bayes", "NB").replace("logistic_regression", "LR")


def restrucutre_data(data, model_path):
    """Utility to process general results.json for easier iteration"""
    model_name = model_path.split("/")[-1].replace(".json", "")
    model_name = short_name(model_name)
    out_acc = [[None for i in range(len(data.keys()))] for j in range(len(data.keys()))]
    out_labels = [[None for i in range(len(data.keys()))] for j in range(len(data.keys()))]
    for i, (dataset_train, data_tests) in enumerate(data.items()):
        for j, (dataset_test, t) in enumerate(data_tests.items()):
            acc, label = max_in_json(t)
            out_acc[i][j] = acc
            out_labels[i][j] = model_name + "\n" + label  # dataset_train + "_" + dataset_test
    return np.array(out_acc), np.array(out_labels)


if __name__ == "__main__":
    all_models = fetch_all_results()
    best_models_results = {model: {"labels": None, "acc": None} for model in all_models}
    for model in all_models:
        with open(model) as f:
            data = json.loads(f.read())
        acc, labels = restrucutre_data(data, model)
        best_models_results[model]["acc"] = acc
        best_models_results[model]["labels"] = labels

    best_result = {"labels": best_models_results[all_models[0]]["labels"],
                   "acc": best_models_results[all_models[0]]["acc"]}

    for _, res in best_models_results.items():
        t_labels = res["labels"]
        t_acc = res["acc"]
        for i in range(len(best_result["acc"])):
            for j in range(len(best_result["acc"])):
                if best_result["acc"][i][j] < t_acc[i][j]:
                    best_result["acc"][i][j] = t_acc[i][j]
                    best_result["labels"][i][j] = t_labels[i][j]
    acc = np.array(best_result["acc"])
    labels = np.array(best_result["labels"])

    fig, ax = plt.subplots()
    # annotate heatmap
    for y in range(acc.shape[0]):
        for x in range(acc.shape[1]):
            ax.text(x + 0.5,
                    y + 0.5,
                    f"{acc[y][x]*100:.1f}%\n{labels[y][x]}",
                    horizontalalignment="center",
                    verticalalignment="center")
    heatmap = ax.pcolor(acc, cmap=CMAP_COLOR_SCHEME, vmax=1, vmin=0)
    ticks = list(data.keys())
    for i, t in enumerate(ticks):
        if t == "matebook14":
            ticks[i] = "mb14"
    ax.set_xticks(np.arange(acc.shape[0]) + 0.5, ticks)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(acc.shape[0]) + 0.5, ticks)
    ax.invert_yaxis()

    # fig.suptitle(f"Best models per (train, test) pair of datasets", size = 13, fontweight = "bold")
    plt.colorbar(heatmap, label="Accuracy")

    if len(sys.argv) == 2:
        if sys.argv[1].split(".")[-1] == "jpg":
            if not os.path.isdir("saved_plots"):
                os.makedirs("saved_plots")
            x_offset = [-55, -30, -50]
        elif sys.argv[1].split(".")[-1] == "pdf":
            if not os.path.isdir("saved_plots_pdf"):
                os.makedirs("saved_plots_pdf")
            x_offset = [-42, -18, -37]

        ax.annotate("________",
                    xy=(0, 1),
                    xytext=(x_offset[0], -5),
                    xycoords="axes fraction",
                    textcoords="offset pixels",
                    rotation=135,
                    fontweight="bold",
                    size=12)
        ax.annotate("Test",
                    xy=(0, 1),
                    xytext=(x_offset[1], 12),
                    xycoords="axes fraction",
                    textcoords="offset pixels",
                    rotation=315,
                    size=12)
        ax.annotate("Train",
                    xy=(0, 1),
                    xytext=(x_offset[2], -7),
                    xycoords="axes fraction",
                    textcoords="offset pixels",
                    rotation=315,
                    size=12)

        plt.savefig(sys.argv[1])
    else:

        ax.annotate("________",
                    xy=(0, 1),
                    xytext=(-42, -5),
                    xycoords="axes fraction",
                    textcoords="offset pixels",
                    rotation=135,
                    fontweight="bold",
                    size=12)
        ax.annotate("Test",
                    xy=(0, 1),
                    xytext=(-17, 20),
                    xycoords="axes fraction",
                    textcoords="offset pixels",
                    rotation=315,
                    size=12)
        ax.annotate("Train",
                    xy=(0, 1),
                    xytext=(-37, -15),
                    xycoords="axes fraction",
                    textcoords="offset pixels",
                    rotation=315,
                    size=12)
        plt.show()
