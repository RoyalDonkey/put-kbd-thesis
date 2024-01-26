"""Module to recreate rnn.json results, based on the pre-trained models"""
import torch
import json
import sys
import os


sys.path.append("../models/rnn/multikeys")
sys.path.append("../pymodules")
import rnn

DATA_ROOT = "../data_balanced/"
VERBOSE = True

OUT_FILE = "results/rnn.json"
ALL_DATASETS = False
MODELS_ROOT = "../rnn_models/"
DATASETS_LIST = ["main_balanced_merged.csv",
                 "g213_balanced_merged_files.csv",
                 "k3p_balanced_merged_files.csv",
                 "matebook14_balanced_merged.csv",
                 "CA_tb14_balanced_merged_files.csv",
                 "all"]


def fetch_all_models():
    files_list = [MODELS_ROOT + x for x in os.listdir(MODELS_ROOT) if x.endswith("pth")]
    return files_list


def iprint(a: str, *args, **kwargs):
    """Helper function to print iff a VERBOSE flag is true"""
    if VERBOSE:
        print(a, *args, **kwargs)


def get_label(s):
    """Derive name of dataset based on file name"""
    return s.replace(".csv", "").replace("balanced", "").replace("_", "").replace("merged", "")


def decode_path(model_path):
    peaks_map = {"thr": "thr",
                 "th":  "th-",
                 "tr":  "t-r",
                 "hr":  "-hr",
                 "t":   "t--",
                 "h":   "-h-",
                 "r":   "--r"}
    model_components = model_path.replace(".pth", "").replace(MODELS_ROOT, "").split("_")
    if len(model_components) > 10:
        dataset = "all"
    else:
        dataset = model_components[1]
    if dataset == "g213" or dataset == "k3p":
        dataset += "files"
    elif dataset == "CA-1406":
        dataset = "CA_tb14files"
    peak = peaks_map[model_components[-8].lower()]
    prep = model_components[-7].lower()
    if prep == "none":
        prep = "raw"
    return dataset, prep, peak


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
    res = {data_training: {data_testing: {prep: {wave: None for wave in wave_comb} for prep in preprocess_comb} for data_testing in dataset_labels.values()} for data_training in dataset_labels.values()}
    models_list = fetch_all_models()

    for i, model_path in enumerate(models_list):
        model = torch.load(model_path, map_location="cpu")
        model = model.to(torch.device("cpu"))
        model.to(torch.device("cpu"))
        print(f"Model {i+1}/{len(models_list)} loaded...")
        training_data, preprocess, wave = decode_path(model_path)
        rnn._config["subset"] = wave.replace("-", "")
        rnn._config["preprocessing"] = model_path.replace(".pth", "").replace(MODELS_ROOT, "").split("_")[-7]
        if rnn._config["preprocessing"] == "None":
            rnn._config["preprocessing"] = None
        for dataset in dataset_files:
            iprint(f"{training_data}\t{dataset_labels[dataset]}\t{preprocess}\t{wave}:\t", end="")
            if dataset != "all":
                acc = rnn.test(model, [dataset_files_full_path[dataset],])
            else:
                acc = rnn.test(model, dataset_files_full_path[dataset])
            iprint(float(acc))
            res[training_data][dataset_labels[dataset]][preprocess][wave] = float(acc)
    # saving scheme: results[training_dataset][testing_dataset][preprocessing][peaks] = acc
    with open(OUT_FILE, "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    iprint(f"Calculating results saved under {OUT_FILE}")
