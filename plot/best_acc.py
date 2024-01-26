#!/usr/bin/env python3
"""Module to plot top N models based on their accuracies. Assuming that if
 classifier was trained on datasetX_train, it was tested on datasetX_test."""
import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

N = 20
BAR_WIDTH = 0.2
DATA_ROOT = "results/"
# if True - gives top results; if False - worst
REVERSE_SORTING = True


def fetch_all_results():
    files_list = [DATA_ROOT + x for x in os.listdir(DATA_ROOT) if x.endswith("json")]
    return files_list


def choose_dataset():
    """Utility to iterate over all possible datasets and allow user to choose one"""
    print("Choose dataset:")
    file_name = fetch_all_results()[0]
    with open(file_name) as f:
        data = json.loads(f.read())
    dataset_labels = list(data.keys())

    for i, dataset_label in enumerate(dataset_labels):
        print(f"\t{i+1}. {dataset_label}")
    print(f"\t{len(dataset_labels) + 1}. Make all plots and save them")
    idx = int(input()) - 1
    assert idx < len(dataset_labels) + 1, f"Input number from 1 to {len(dataset_labels) + 1}"
    if idx == len(dataset_labels):
        print("Saving all the plots...")
        return None
    else:
        print(f"Chosen model: {dataset_labels[idx]}")
        return dataset_labels[idx]


def find_best_acc(data, dataset="all"):
    """Utility to process general results.json for easier iteration"""
    data = data[dataset][dataset]
    out = []
    for preprocess, peaks in data.items():
        for peak, acc in peaks.items():
            out.append([f"_{preprocess}_{peak}", acc])
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:N]


def plot_routine(chosen_dataset):
    files_list = fetch_all_results()
    best_acc_list = []
    for file in files_list:
        with open(file) as f:
            data = json.loads(f.read())
            data = find_best_acc(data, chosen_dataset)
            for i in range(len(data)):
                data[i][0] = file.replace(".json", "").replace(DATA_ROOT, "") + data[i][0]
            best_acc_list += data
    best_acc_list.sort(key=lambda x: x[1], reverse=REVERSE_SORTING)
    best_acc_list = best_acc_list[:N]

    models = [label for label, acc in best_acc_list]
    labels = [model.split("_")[-2] for model in models]
    y = [acc for label, acc in best_acc_list]
    y_pos = [i for i in range(len(models))]
    color_map = {"raw": "tab:blue", "fft": "tab:orange", "mfcc": "tab:green"}
    colors = [color_map[l] for l in labels]

    fig, ax = plt.subplots(figsize=[8.0, 4.8])
    ax.barh(y_pos, y, color=colors)
    ax.invert_yaxis()
    ax.set_yticks(y_pos, labels=models)
    ax.set_xlim([0, 1])
    ax.set_xlabel("Accuracy", size=12)

    # ax.set_title(f"Best models, by accuracy,\ntrained on {chosen_dataset}", size=16, fontweight="bold")
    fig.subplots_adjust(left=0.3, top=0.9, bottom=0.1, right=0.9)
    handles, labels = ax.get_legend_handles_labels()
    handles = [Patch(color=v, label=k) for k, v in color_map.items()]
    plt.legend(title="Preprocessing",
               labels=["raw", "fft", "mfcc"],
               handles=handles,
               # loc = "lower right",
               # bbox_to_anchor=(1.05, 0.0)
               )


if __name__ == "__main__":
    chosen_dataset = choose_dataset()
    if chosen_dataset is not None:
        plot_routine(chosen_dataset)
        plt.show()
    else:
        file = fetch_all_results()[0]
        with open(file) as f:
            data = json.loads(f.read())
        dataset_labels = list(data.keys())
        print("Choose a format to save:")
        print("\t1. jpg")
        print("\t2. pdf")
        decision = input()
        if int(decision) == 1:
            if not os.path.isdir("saved_plots/best_acc_per_dataset"):
                os.makedirs("saved_plots/best_acc_per_dataset")
            for label in dataset_labels:
                plot_routine(label)
                plt.savefig("saved_plots/best_acc_per_dataset/" + label + "_dataset.jpg")
        elif int(decision) == 2:
            if not os.path.isdir("saved_plots_pdf/best_acc_per_dataset"):
                os.makedirs("saved_plots_pdf/best_acc_per_dataset")
            for label in dataset_labels:
                plot_routine(label)
                plt.savefig("saved_plots_pdf/best_acc_per_dataset/" + label + "_dataset.pdf")
        else:
            print("Please press a number corresponding to your choice")
