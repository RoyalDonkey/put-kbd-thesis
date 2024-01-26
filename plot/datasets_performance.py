#!/usr/bin/env python3
"""Module to plot results of particular classifier, over all datasets,
preprocessings and peaks."""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_ROOT = "results/"
NROWS, NCOLS = 2, 3
BAR_WIDTH = 0.2
CMAP_COLOR_SCHEME = "Spectral"


def fetch_all_results():
    files_list = [DATA_ROOT + x for x in os.listdir(DATA_ROOT) if x.endswith("json")]
    return files_list


def get_name(s):
    return s.replace(".json", "").replace("results/", "")


def choose_model():
    """Utility to iterate over DATA_ROOT files and allow user to choose one"""
    print("Choose dataset:")
    files_name = sorted(fetch_all_results())
    models_name = [get_name(s) for s in files_name]

    for i, dataset_label in enumerate(models_name):
        print(f"\t{i+1}. {dataset_label}")
    print(f"\t{len(models_name) + 1}. Make all plots and save them")
    idx = int(input()) - 1
    assert idx < len(models_name) + 1, f"Input number from 1 to {len(models_name) + 1}"
    if idx == len(models_name):
        print("Saving all the plots...")
        return None
    else:
        print(f"Chosen model: {models_name[idx]}")
        return files_name[idx]


def max_in_json(data):
    mx = 0
    mx_label = "NOT_FOUND"
    for prep, peaks in data.items():
        for peak, acc in peaks.items():
            if acc > mx:
                mx = acc
                mx_label = f"{prep}_{peak}"
    return mx, mx_label


def restrucutre_data(data):
    """Utility to process general results.json for easier iteration"""
    out_acc = [[None for i in range(len(data.keys()))] for j in range(len(data.keys()))]
    out_labels = [[None for i in range(len(data.keys()))] for j in range(len(data.keys()))]
    for i, (dataset_train, data_tests) in enumerate(data.items()):
        for j, (dataset_test, t) in enumerate(data_tests.items()):
            acc, label = max_in_json(t)
            out_acc[i][j] = acc
            out_labels[i][j] = label  # dataset_train + "_" + dataset_test
    return np.array(out_acc), np.array(out_labels)


def plot_routine(chosen_model, x_offset):
    with open(chosen_model) as f:
        data = json.loads(f.read())
    acc, labels = restrucutre_data(data)

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
    ax.set_xticks(np.arange(len(data.keys())) + 0.5, ticks)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(data.keys())) + 0.5, ticks)
    ax.invert_yaxis()

    if x_offset == ".jpg":
        x_offset = [-55, -35, -50]
    elif x_offset == ".pdf":
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
    # model_name = get_name(chosen_model)
    # fig.suptitle(f"Best results of architecture {model_name} over datasets", size=14, weight="bold")

    plt.colorbar(heatmap, label="Accuracy")


if __name__ == "__main__":
    chosen_model = choose_model()
    if chosen_model is not None:
        plot_routine(chosen_model, x_offset=".jpg")
        plt.show()
    else:
        files_name = sorted(fetch_all_results())
        print("Choose a format to save:")
        print("\t1. jpg")
        print("\t2. pdf")
        decision = input()
        if int(decision) == 1:
            if not os.path.isdir("saved_plots/dataset_performances"):
                os.makedirs("saved_plots/dataset_performances")
            for model in files_name:
                plot_routine(model, x_offset=".jpg")
                plt.savefig("saved_plots/dataset_performances/" + get_name(model) + ".jpg")
        elif int(decision) == 2:
            if not os.path.isdir("saved_plots_pdf/dataset_performances"):
                os.makedirs("saved_plots_pdf/dataset_performances")
            for model in files_name:
                plot_routine(model, x_offset=".pdf")
                plt.savefig("saved_plots_pdf/dataset_performances/" + get_name(model) + ".pdf")
        else:
            print("Please press a number corresponding to your choice")
