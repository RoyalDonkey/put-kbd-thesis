#!/usr/bin/env python3
"""Module to iterate over all testing, training datasets, all preprocessings and peaks
to create .json file with all accuracies. Mainly use to create summary of accuracies,
and plot them with other scripts in this module. You need to adjust small parts for
your model."""
import json
import sys
import os


sys.path.append("../pymodules")
sys.path.append("../models/rnn/multikeys")
import rnn

DATA_ROOT = "../data_balanced/"
VERBOSE = True
SHUFFLE_DATA = False
OUT_FILE = "results/rnn.json"
ALL_DATASETS = False
DATASETS_LIST = ["main_balanced_merged.csv",
                 "g213_balanced_merged.csv",
                 "k3p_balanced_merged.csv",
                 "matebook14_balanced_merged.csv",
                 "CA_tb14_balanced_merged.csv",
                 "all"]
EPOCHS = 20


def iprint(a: str, *args, **kwargs):
    """Helper function to print iff a VERBOSE flag is true"""
    if VERBOSE:
        print(a, *args, **kwargs)


def get_label(s: str) -> str:
    """Derive name of dataset based on file name"""
    return s.replace(".csv", "").replace("balanced", "").replace("_", "").replace("merged", "")


if __name__ == "__main__":
    iprint("Loading data...")

    if ALL_DATASETS:
        dataset_files = [DATA_ROOT + x for x in os.listdir(DATA_ROOT) if x.endswith(".csv")]
    else:
        dataset_files = DATASETS_LIST

    dataset_labels = {file_name: get_label(file_name) for file_name in dataset_files}
    dataset_files_full_path = {f: DATA_ROOT + f for f in dataset_files}
    dataset_files_full_path["all"] = [DATA_ROOT + f for f in dataset_files if f != "all"]

    preprocess_comb = ["raw", "fft", "mfcc"]
    wave_comb = ["thr", "th-", "t-r", "-hr", "t--", "-h-", "--r"]
    results = {data_training: {data_testing: {prep: {wave: None for wave in wave_comb} for prep in preprocess_comb} for data_testing in dataset_labels.values()} for data_training in dataset_labels.values()}
    for training_data in dataset_files:
        iprint(f"Starting to train on dataset {training_data}:")
        for preprocess in preprocess_comb:
            iprint(f"\twith preprocessing {preprocess}:")
            for wave in wave_comb:
                iprint(f"\t\t{wave}, against dataset:")
                if preprocess == "raw":
                    rnn._config["preprocessing"] = None
                if preprocess == "mfcc":
                    rnn._config["preprocessing"] = "MFCC"
                if preprocess == "fft":
                    rnn._config["preprocessing"] = "FFT"
                rnn._config["subset"] = wave.replace("-", "")
                rnn._config["epochs"] = EPOCHS
                if training_data != "all":
                    model = rnn.train([dataset_files_full_path[training_data],])
                else:
                    model = rnn.train(dataset_files_full_path[training_data])
                for dataset in dataset_files:
                    if dataset != "all":
                        acc = rnn.test(model, [dataset_files_full_path[dataset],])
                    else:
                        acc = rnn.test(model, dataset_files_full_path[dataset])
                # =====================
                    iprint(f"\t\t\t{dataset_labels[dataset]}:\t{acc}")
                    results[dataset_labels[training_data]][dataset_labels[dataset]][preprocess][wave] = acc
    # saving scheme: results[training_dataset][testing_dataset][preprocessing][peaks] = acc
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    iprint(f"Calculating results saved under {OUT_FILE}")
