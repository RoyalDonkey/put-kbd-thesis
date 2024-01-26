#!/usr/bin/env python3
"""Module to plot results of particular classifier, over all datasets,
preprocessings and peaks. Assuming that if classifier was trained on datasetX_train,
it was tested on datasetX_test."""
import json
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_ROOT = "results/"
NROWS, NCOLS = 2, 3
BAR_WIDTH = 0.2


def fetch_all_results():
    files_list = [DATA_ROOT + x for x in os.listdir(DATA_ROOT) if x.endswith("json")]
    return files_list


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


def restrucutre_data(data):
    """Utility to process general results.json for easier iteration"""
    for dataset in data.keys():
        data[dataset] = data[dataset][dataset]
    out = {dataset: {prep: [] for prep in preprocessings.keys()} for dataset, preprocessings in data.items()}
    for dataset, preprocessings in data.items():
        for prep, peaks in preprocessings.items():
            for peak, acc in peaks.items():
                out[dataset][prep].append((peak, acc))
    # sort peaks
    for dataset, preprocessings in out.items():
        for peak, acc in preprocessings.items():
            out[dataset][peak].sort(key=lambda x: x[0], reverse=True)
            out[dataset][peak] = [acc for peak, acc in out[dataset][peak]]
    return out


def get_name(s):
    return s.replace(".json", "").replace("results/", "")


def plot_routine(chosen_model):
    with open(chosen_model) as f:
        data = json.loads(f.read())
    x_ticks = sorted(["thr", "th-", "t-r", "-hr", "t--", "-h-", "--r"], reverse=True)
    x_ticks = [tick.replace("-", "") for tick in x_ticks]
    data = restrucutre_data(data)
    fig, ax = plt.subplots(nrows=NROWS, ncols=NCOLS, sharex=True, sharey=True)
    for i, (dataset, preprocessings) in enumerate(data.items()):
        i_ax = i // NCOLS
        j_ax = i % NCOLS
        for j, (prep, acc_list) in enumerate(preprocessings.items()):
            x = np.arange(len(acc_list))
            x = x - (BAR_WIDTH * (j - (len(preprocessings.keys()) // 2)))
            ax[i_ax, j_ax].bar(x, acc_list, BAR_WIDTH, label=prep)
        ax[i_ax, j_ax].set_title(f"{dataset}")
        ax[i_ax, j_ax].set_ylim([0, 1.0])
        ax[i_ax, j_ax].set_xticks(range(len(x_ticks)), x_ticks)
        handles, labels = ax[i_ax, j_ax].get_legend_handles_labels()
    # model_name = chosen_model.replace("results/", "").replace(".json", "")
    # fig.suptitle(f"Accuracies of {model_name} on different datasets", fontweight="bold")
    fig.supxlabel("Peaks")
    fig.supylabel("Accuracy")
    fig.subplots_adjust(left=0.15, top=0.88, bottom=0.1, right=0.9)
    fig.legend(handles, labels, title="Preprocessing", framealpha=0.3)


if __name__ == "__main__":
    chosen_model = choose_model()
    if chosen_model is not None:
        plot_routine(chosen_model)
        plt.show()
    else:
        files_name = sorted(fetch_all_results())
        print("Choose a format to save:")
        print("\t1. jpg")
        print("\t2. pdf")
        decision = input()
        if int(decision) == 1:
            if not os.path.isdir("saved_plots/model_summary"):
                os.makedirs("saved_plots/model_summary")
            for model in files_name:
                plot_routine(model)
                plt.savefig("saved_plots/model_summary/" + get_name(model) + ".jpg")
        elif int(decision) == 2:
            if not os.path.isdir("saved_plots_pdf/model_summary"):
                os.makedirs("saved_plots_pdf/model_summary")
            for model in files_name:
                plot_routine(model)
                plt.savefig("saved_plots_pdf/model_summary/" + get_name(model) + ".pdf")
        else:
            print("Please press a number corresponding to your choice")
