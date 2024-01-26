#!/usr/bin/env python3
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from joblib import dump  # noqa: F401
from itertools import combinations

import sys
sys.path.append("../../pymodules")
from custom_read_csv import KeyPressSequence
from visualizations import prediction_heatmap
from fft_tools import new_fft
from mfcc_tools import mfcc


def make_alphabet(kps: KeyPressSequence) -> list[str]:
    """Create an ordered list of keys in kps (no duplicates)"""
    alphabet = set()
    for kpd in kps.data:
        alphabet.add(kpd.key)
    return sorted(list(alphabet))


def raw_shuffled_data(main_kps: KeyPressSequence,
                      test_sample_ratio: float = 1 / 11, random_seed: int = 0
                      ) -> tuple[KeyPressSequence, KeyPressSequence]:
    """Given a KeyPressSequence, return its partition into two parts - a training,
    and a testing KeyPressSequence.
    `test_sample_ratio` controls what part of all samples of a character are
    assigned to the test set.
    """
    counting_dict = dict()
    for kpd in main_kps.data:
        if kpd.key in counting_dict.keys():
            counting_dict[kpd.key] += 1
        else:
            counting_dict[kpd.key] = 1
    test_count_dict = dict(zip(counting_dict.keys(),
                               map(lambda x: int(x * test_sample_ratio),
                               counting_dict.values())))

    np.random.seed(random_seed)
    np.random.shuffle(main_kps.data)
    test_kps = KeyPressSequence()
    train_kps = KeyPressSequence()
    for kpd in main_kps:
        if test_count_dict[kpd.key] > 0:
            test_kps.data.append(kpd)
            test_count_dict[kpd.key] -= 1
        else:
            train_kps.data.append(kpd)
    return train_kps, test_kps


def data_peak_type(kps: KeyPressSequence, peak_type: str,
                   preprocessing: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    df = kps.to_pandas()
    peak_subset = peak_type.lower()
    assert set(peak_subset) <= set("thr"), \
        f"Illegal peak type: {peak_type}; a subset of 'thr' is expected"
    assert preprocessing in {None, "FFT", "MFCC"}, \
        f"Illegal preprocessing technique: {preprocessing}; expected None, 'FFT', or 'MFCC'"
    col_names = [{"t": "touch", "h": "hit", "r": "release"}[c] for c in peak_subset]
    Y = np.array(df["key"])
    df = df[col_names]
    match preprocessing:
        case None:
            pass
        case "FFT":
            for col_name in col_names:
                df[col_name] = df[col_name].map(new_fft)
        case "MFCC":
            for col_name in col_names:
                df[col_name] = df[col_name].map(mfcc)

    X = []
    for _, row in df.iterrows():
        to_append = []
        for c in col_names:
            to_append += row[c].tolist()
        X.append(to_append)

    X = np.array(X).astype(np.float32)
    if preprocessing != "MFCC":
        X /= 2 ** 15  # LinearRegression asked for scaling or increasing max_iter
        # no suitable number of iterations was found, so this was chosen as the fix
    if preprocessing == "MFCC":
        # LR doesn't accept 3D data. This reshapes it from (num_samples, num_peaks, 16) to (num_samples, num_peaks*16)
        X = X.reshape(-1, X.shape[-2] * X.shape[-1])
    return X, Y


def logistic_regression_peak_type(kps: KeyPressSequence, peak_type: str,
                                  preprocessing: str | None) -> LogisticRegression:
    X, Y = data_peak_type(kps, peak_type, preprocessing)
    lr_classifier = LogisticRegression(random_state=0, multi_class="ovr", max_iter=10000)
    lr_classifier = lr_classifier.fit(X, Y)
    return lr_classifier


if __name__ == "__main__":
    PREPROCESSING: None | str = "FFT"
    # if set to false - test data will come from sessions starting with "test"
    # if set to true - test data is randomly selected from all examples
    # see `raw_shuffled_data()` for implementation
    SHUFFLE_SESSIONS: bool = False

    # PEAK_TYPE = "thr"
    thr_combinations = [list(combinations("thr", i)) for i in range(1, 4)]
    thrs = []
    for combos in thr_combinations:
        for t in combos:
            thrs.append("".join(t))
    for PEAK_TYPE in thrs:
        EXPORT_FILENAME = f"logistic_regression_{PEAK_TYPE}_"
        if PREPROCESSING is None:
            EXPORT_FILENAME += "raw"
        else:
            EXPORT_FILENAME += PREPROCESSING
        EXPORT_FILENAME += ".joblib"
        main_kps = KeyPressSequence("../../data_balanced/main_balanced_merged.csv")
        train_kps = KeyPressSequence()
        test_kps = KeyPressSequence()
        if not SHUFFLE_SESSIONS:
            for kpd in main_kps.data:
                if kpd.session[:4] == "test":
                    test_kps.data.append(kpd)
                else:
                    train_kps.data.append(kpd)
        if SHUFFLE_SESSIONS:
            train_kps, test_kps = raw_shuffled_data(main_kps)
        x, y = data_peak_type(train_kps, PEAK_TYPE, PREPROCESSING)
        classifier = logistic_regression_peak_type(train_kps, PEAK_TYPE, PREPROCESSING)
        test_X, test_Y = data_peak_type(test_kps, PEAK_TYPE, PREPROCESSING)
        test_predictions = classifier.predict(test_X)

        print("Accuracy:", accuracy_score(test_Y, test_predictions))
        alphabet = make_alphabet(train_kps)
        prediction_heatmap(test_predictions, test_Y, alphabet, f"Peak Type: {PEAK_TYPE}\nPREPROCESSING: {PREPROCESSING}")
        # dump(classifier, EXPORT_FILENAME)
