"""Custom plotting script for comparing different models on one dataset in similar fashion to "model_comparison.py" """
# !/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import os


DATA_ROOT = "results/"
FIXED_PEAK = "thr"
FIXED_Y = True
NCOLS = 2
BAR_WIDTH = 0.2
DATASET = "CA_tb14"
MODEL_NAMES = ["naive_bayes", "svm"]
MODELS = [DATA_ROOT + m + ".json" for m in MODEL_NAMES]


def fetch_all_results():
    files_list = [DATA_ROOT + x for x in os.listdir(DATA_ROOT) if x.endswith("json")]
    return files_list


def restructure_data(data, peak=FIXED_PEAK):
    """Utility to process general results.json for easier iteration"""
    for dataset in data.keys():
        data[dataset] = data[dataset][dataset]
    for dataset in sorted(data.keys()):
        for prep in data[dataset].keys():
            data[dataset][prep] = data[dataset][prep][FIXED_PEAK]
    return data


def plot_routine(models=MODELS):
    files_list = sorted(fetch_all_results())
    fig, ax = plt.subplots(ncols=NCOLS, sharex=True, sharey=True)
    i_ax = 0
    for file_name in files_list:
        if file_name not in models:
            continue
        with open(file_name) as f:
            data = json.loads(f.read())
        data = restructure_data(data)[DATASET]
        print(data)
        for q, (k, v) in enumerate(data.items()):
            ax[i_ax].bar(k, v, label=k)
        title = file_name.split("/")[-1].replace(".json", "")
        ax[i_ax].set_title(f"{title}")
        # ax[i_ax].set_ylim([0, 1])
        handles, labels = ax[i_ax].get_legend_handles_labels()
        # fig.supxlabel("Preprocesses")
        # fig.supylabel("Accuracy")
        # fig.subplots_adjust(left=0.15, top=0.8, bottom=0.1, right=0.9)
        # fig.legend(handles, labels, loc="upper left", title="Preprocessing", framealpha=0.3)
        i_ax += 1


if __name__ == "__main__":
    # Default size is (6.4, 4.8); values in inches
    if NCOLS == 2:
        plt.rcParams["figure.figsize"] = (3.2, 2.4)
    elif NCOLS == 3:
        plt.rcParams["figure.figsize"] = (4.2, 2.4)
    plot_routine()
    plt.savefig(f"saved_plots_pdf/model_comparison/custom_{'_'.join(MODEL_NAMES)}_{DATASET}.pdf")
