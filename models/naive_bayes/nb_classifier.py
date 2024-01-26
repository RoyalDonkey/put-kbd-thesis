#!/usr/bin/env python3
"""Module with basic usage of Naive Bayes classifier. Also compares result for
different options such that different peaks, fft and relaxed accuracy (described below)"""
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.append("../../pymodules")
import custom_read_csv as crc
from build_kps import build_kps
from fft_tools import new_fft
DATA_ROOT = "../../data/"
RELAXED_CUT = 0.90
VERBOSE = True


def iprint(a: str, *args, **kwargs):
    """Helper function to print iff a VERBOSE flag is true"""
    if VERBOSE:
        print(a, *args, **kwargs)


def relaxed_accuracy(
    y_log_pred: np.ndarray[np.ndarray],
    y_test: np.ndarray[str],
    classes: np.ndarray[str],
    cutoff: float = RELAXED_CUT
) -> float:
    """Calculate relaxed accuracy of of results, by checking if true label is
    in all characters for which classifier assigned value at least as big as
    cutoff * maximal_score_of_predictions_of_all_characters
    (basically check if e.g. true_label = "a" in relaxed_label = "abc" )"""
    assert 0 <= RELAXED_CUT <= 1, "illegal cutoff value"
    assert len(classes) > 0, "too low number of classes"
    y_log_pred = [normalize(arr) for arr in y_log_pred]
    max_values = [np.max(arr) for arr in y_log_pred]
    pred_labeled = [[[ch, value] for value, ch in zip(l, classes)] for l in y_log_pred]
    y_pred_relaxed = []
    for mx, arr in zip(max_values, pred_labeled):
        y_pred_relaxed.append(find_relaxed_labels(arr, cutoff))
    acc_t = np.sum(np.array([test in pred for pred, test in zip(y_pred_relaxed, y_test)])) / np.size(y_test)
    iprint(f"\t\t\tAve num of char for settings underneath: {np.average([len(s) for s in y_pred_relaxed]):.3f}")
    return np.sum(np.array(acc_t))


def normalize(arr: np.array) -> np.array:
    """Normalize arr to range between <0; 1>"""
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def find_relaxed_labels(arr: np.ndarray, cutoff: float) -> str:
    """Helper functions to create a string of characters,
    for which value is greater or equal to cutoff"""
    return "".join([ch if value >= cutoff else "" for ch, value in arr])


def tiny_fft(arr: np.ndarray) -> np.ndarray:
    """Construct arr of values via fft transformation"""
    fft_arr = np.fft.fft(arr)
    fft_arr = np.abs(fft_arr)
    n = len(fft_arr) // 2
    y = 2 * fft_arr[:n] / n
    return y


def join_row(row: np.array) -> pd.Series:
    """Helper function to concatenate values,
    of all lists from row into one pd.Series"""
    out = np.array([])
    for cell in row:
        out = np.concatenate([out, cell], axis=0)
    return pd.Series(out)


def data_loader(
    kps: crc.KeyPressSequence,
    fft: bool = False,
    subset: Optional[str] = None
) -> tuple[np.array, np.array]:
    """Transform provided kps to easier to iterate X and y np.array,
    based on provided subset and fft flag"""
    df = kps.to_pandas()
    if subset is not None:
        assert set(subset) <= set("thr"), "illegal subset value"
        assert len(subset) != 0, "subset cannot be empty"
        col_names = ["key"] + [{"t": "touch", "h": "hit", "r": "release"}[c] for c in subset]
        df = df[col_names]
    features_names = list(df.columns)
    features_names.remove("key")
    # concatenate all columns into one with one list,
    # depending on provided subset
    X = []
    for i, row in df.iterrows():
        t = []
        for name in features_names:
            t += row[name].tolist()
        X.append(t.copy())
        del t
    if fft:
        for i, row in enumerate(X):
            X[i] = new_fft(row)
    y = df["key"].to_numpy()
    return np.array(X), y


if __name__ == "__main__":
    """Conduct experiments, testing all combinations of peaks,
    fft and cutout"""
    kps_train = build_kps(DATA_ROOT, "train")
    kps_test = build_kps(DATA_ROOT, "test")
    all_combinations = ["thr", "th", "tr", "hr", "t", "h", "r"]
    all_fft_and_relaxed = [[True, False],
                           [False, False],
                           [False, True],
                           [True, True]]
    results = [[None for i in all_combinations] for j in all_fft_and_relaxed]
    iprint("Data loaded, conducting experiments...\n")
    # iterate over all options
    for i, comb in enumerate(all_combinations):
        for j, (fft, log_prob) in enumerate(all_fft_and_relaxed):
            # setup classifier and data
            X_train, y_train = data_loader(kps_train, fft, comb)
            X_test, y_test = data_loader(kps_test, fft, comb)
            gnb = GaussianNB()
            # calculate accuracy
            if log_prob:
                y_log_pred = gnb.fit(X_train, y_train).predict_log_proba(X_test)
                acc = relaxed_accuracy(y_log_pred, y_test, gnb.classes_)
            else:
                y_pred = gnb.fit(X_train, y_train).predict(X_test)
                acc = accuracy_score(y_test, y_pred)
            iprint(f"For {comb}, {fft=}, {log_prob=} accuracy is equal to:  \t{acc:.3f}")
            results[j][i] = acc
    # plotting results of experiments
    fig, ax = plt.subplots()
    im = ax.imshow(results, cmap="summer_r")
    ax.set_xticks(np.arange(len(all_combinations)), labels=all_combinations)
    ax.set_yticks(np.arange(len(all_fft_and_relaxed)), labels=all_fft_and_relaxed)
    ax.set_title("Naive Bayes Classifier accuracies")
    ax.set_ylabel('[fft, relaxed_acc]')
    ax.set_xlabel('Peaks combination')
    for i in range(len(all_fft_and_relaxed)):
        for j in range(len(all_combinations)):
            text = ax.text(j, i, f"{results[i][j]:.3f}", ha="center", va="center", color="black")
    fig.tight_layout()
    plt.show()
