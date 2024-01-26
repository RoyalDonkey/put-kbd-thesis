#!/usr/bin/env python3
"""Module to plot comparisons between models, based on particular
dataset. Assuming that if classifier was trained on datasetX_train,
it was tested on datasetX_test."""
import json
import matplotlib.pyplot as plt
import os


DATA_ROOT = "results/"
FIXED_PEAK = "thr"
FIXED_Y = True
NROWS, NCOLS = 2, 3
BAR_WIDTH = 0.2


def fetch_all_results():
    files_list = [DATA_ROOT + x for x in os.listdir(DATA_ROOT) if x.endswith("json")]
    return files_list


def choose_dataset() -> str | None:
    """Utility to iterate over all possible datasets and allow user to choose one"""
    print("Choose dataset:")
    file_name = fetch_all_results()[0]
    with open(file_name) as f:
        data = json.loads(f.read())
    dataset_labels: list[str] = list(data.keys())

    for i, dataset_label in enumerate(dataset_labels):
        print(f"\t{i+1}. {dataset_label}")
    print(f"\t{len(dataset_labels) + 1}. Make all plots and save them")
    idx = int(input()) - 1
    assert idx < len(dataset_labels) + 1, f"Input number from 1 to {len(dataset_labels) + 1}"
    if idx == len(dataset_labels):
        print("Saving all the plots...")
        return None
    else:
        print(f"Chosen dataset: {dataset_labels[idx]}")
        return dataset_labels[idx]


def restructure_data(data, peak=FIXED_PEAK):
    """Utility to process general results.json for easier iteration"""
    for dataset in data.keys():
        data[dataset] = data[dataset][dataset]
    for dataset in sorted(data.keys()):
        for prep in data[dataset].keys():
            data[dataset][prep] = data[dataset][prep][FIXED_PEAK]
    return data


def get_ylimit(files_list, chosen_dataset):
    """Flexible ylimit which is 1.25 of maximum value found in results
    files with fixed dataset.
    """
    mx = 0.0
    for i, file_name in enumerate(files_list):
        with open(file_name) as f:
            data = json.loads(f.read())
        data = restructure_data(data)[chosen_dataset]
        for k, v in data.items():
            if v > mx:
                mx = v
    if FIXED_Y:
        return 1.0
    else:
        return min(mx * 1.25, 1.0)


def plot_routine(chosen_dataset):
    files_list = sorted(fetch_all_results())
    fig, ax = plt.subplots(nrows=NROWS, ncols=NCOLS, sharex=True, sharey=True)
    y_limit = get_ylimit(files_list, chosen_dataset)
    for i, file_name in enumerate(files_list):
        with open(file_name) as f:
            data = json.loads(f.read())
        data = restructure_data(data)[chosen_dataset]
        print(file_name, data)
        i_ax = i // NCOLS
        j_ax = i % NCOLS
        for q, (k, v) in enumerate(data.items()):
            ax[i_ax, j_ax].bar(k, v, label=k)
        title = file_name.split("/")[-1].replace(".json", "")
        ax[i_ax, j_ax].set_title(f"{title}")
        ax[i_ax, j_ax].set_ylim([0, y_limit])
        handles, labels = ax[i_ax, j_ax].get_legend_handles_labels()
    # fig.suptitle(f"Model comparison,\n{FIXED_PEAK} + {chosen_dataset}", fontweight="bold", size=14)
    fig.supylabel("Accuracy")
    fig.subplots_adjust(left=0.15, top=0.8, bottom=0.1, right=0.9)
    fig.legend(handles, labels, loc="upper left", title="Preprocessing", framealpha=0.3)


if __name__ == "__main__":
    chosen_dataset = choose_dataset()
    if chosen_dataset is not None:
        plot_routine(chosen_dataset)
        plt.show()
    else:
        file_name = fetch_all_results()[0]
        with open(file_name) as f:
            data = json.loads(f.read())
        dataset_labels = list(data.keys())
        print("Choose a format to save:")
        print("\t1. jpg")
        print("\t2. pdf")
        decision = input()
        if int(decision) == 1:
            if not os.path.isdir("saved_plots/model_comparison"):
                os.makedirs("saved_plots/model_comparison")
            for label in dataset_labels:
                plot_routine(label)
                plt.savefig(f"saved_plots/model_comparison/on_{label}_dataset.jpg")
        elif int(decision) == 2:
            if not os.path.isdir("saved_plots_pdf/model_comparison"):
                os.makedirs("saved_plots_pdf/model_comparison")
            for label in dataset_labels:
                plot_routine(label)
                plt.savefig(f"saved_plots_pdf/model_comparison/on_{label}_dataset.pdf")
        else:
            print("Please press a number corresponding to your choice")
