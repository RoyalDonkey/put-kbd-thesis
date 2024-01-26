#!/usr/bin/env python3
"""Module to iterate over all testing, training datasets, all preprocessings and peaks
to create .json file with all accuracies. Mainly use to create summary of accuracies,
and plot them with other scripts in this module. You need to adjust small parts for
your model."""
from sklearn.naive_bayes import GaussianNB  # noqa: F401
from sklearn.neighbors import KNeighborsClassifier  # noqa: F401
from sklearn import svm  # noqa: F401
from sklearn.linear_model import LogisticRegression  # noqa: F401
from sklearn.metrics import accuracy_score
from typing import Optional
import numpy as np
import json
import sys
import os


sys.path.append("../pymodules")
from custom_read_csv import KeyPressSequence
from fft_tools import new_fft
from mfcc_tools import mfcc

sys.path.append("../models/xgboost")
import xgboost_results_wrapper  # noqa: F401


DATA_ROOT = "../data_balanced/"
VERBOSE = True
SHUFFLE_DATA = False
OUT_FILE = "results/__YOUR__MODEL_NAME__.json"
ALL_DATASETS = False
DATASETS_LIST = ["main_balanced_merged.csv",
                 "matebook14_balanced_merged.csv",
                 "g213_balanced_merged.csv",
                 "k3p_balanced_merged.csv",
                 "CA_tb14_balanced_merged.csv",
                 "all"]
N_KEYS = 43
TRAIN_SIZE = 100
TEST_SIZE = 10
SAMPLE_SIZE = {
    "raw": 176,
    "fft": 176,
    "mfcc": 16,
}


def iprint(a: str, *args, **kwargs):
    """Helper function to print iff a VERBOSE flag is true"""
    if VERBOSE:
        print(a, *args, **kwargs)


def shuffle_datasets(train_data, test_data, ratio=1 / 11):
    """Shuffle train and test datasets from all sessions and derive
    new train and test sets, according to ratio"""
    RANDOM_SEED = 0
    main_kps = train_data + test_data
    counting_dict = dict()
    for kpd in main_kps.data:
        if kpd.key in counting_dict.keys():
            counting_dict[kpd.key] += 1
        else:
            counting_dict[kpd.key] = 1
    test_count_dict = dict(zip(counting_dict.keys(),
                               map(lambda x: int(x * ratio),
                               counting_dict.values())))

    np.random.seed(RANDOM_SEED)
    np.random.shuffle(main_kps.data)
    test_kps = KeyPressSequence()
    train_kps = KeyPressSequence()
    for kpd in main_kps:
        if test_count_dict[kpd.key] > 0:
            test_kps.data.append(kpd)
            test_count_dict[kpd.key] -= 1
        else:
            train_kps.data.append(kpd)
    d = dict()
    for kpd in train_kps.data:
        if kpd.key in d.keys():
            d[kpd.key] += 1
        else:
            d[kpd.key] = 1
    return train_kps, test_kps


def data_loader(
    kps: KeyPressSequence,
    fft_flag: bool = False,
    magnitude: bool = True,
    subset: Optional[str] = None,
    mfcc_flag: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Transform provided kps to easier to iterate X and y np.array,
    based on provided subset and other flags"""
    df = kps.to_pandas()
    assert (mfcc_flag and not fft_flag) or (not mfcc_flag and fft_flag) or (not mfcc_flag and not fft_flag), "only fft or mfcc at once"
    if subset is not None:
        assert set(subset) <= set("thr"), "illegal subset value"
        assert len(subset) != 0, "subset cannot be empty"
        col_names = ["key"] + [{"t": "touch", "h": "hit", "r": "release"}[c] for c in subset]
        df = df[col_names]
    feature_names = list(df.columns)
    feature_names.remove("key")

    # apply preprocessing for each peak separately
    if fft_flag:
        for feature_name in feature_names:
            # apply is necessary to provide keyword arguments
            df[feature_name] = df[feature_name].apply(new_fft, magnitude=magnitude)
    if mfcc_flag:
        for feature_name in feature_names:
            df[feature_name] = df[feature_name].map(mfcc)
    # tmp is an array of tuples - each tuple holds arrays
    # corresponding to each of the features/peaks
    # : - select all rows, feature_names - list of columns to select from rows
    tmp = df.loc[:, feature_names].to_numpy()
    # concatenate all columns into one with one list, depending on provided subset
    X = np.array([np.concatenate(feature) for feature in tmp])
    y = df["key"].to_numpy()
    # ONLY FOR LINEAR REGRESSION
    # if not mfcc_flag:
    #    X = np.array(X).astype(np.float32)
    #    X /= 2 ** 15
    if mfcc_flag:
        X = np.array(X)
        X = X.reshape(-1, X.shape[-2] * X.shape[-1])
    return np.array(X), y


def build_train_test(balanced_data_file: str, shuffle_data: bool = False):
    data_temp = KeyPressSequence(DATA_ROOT + balanced_data_file)
    kps_train = KeyPressSequence()
    kps_test = KeyPressSequence()
    for e in data_temp.data:
        if "train" in e.session.split("_")[0]:
            kps_train.data.append(e)
        elif "test" in e.session.split("_")[0]:
            kps_test.data.append(e)
    if shuffle_data:
        kps_train, kps_test = shuffle_datasets(kps_train, kps_test)
    return kps_train, kps_test


def build_from_all(files_list: list, shuffle_data: bool = False):
    """Utility function to merge all data from files_list into one big dataset"""
    files_list = [DATA_ROOT + file for file in files_list if file != "all"]
    kps_train = KeyPressSequence()
    kps_test = KeyPressSequence()
    for file in files_list:
        if file == DATA_ROOT + "all":
            continue
        data_temp = KeyPressSequence(file)
        for e in data_temp.data:
            if "train" in e.session.split("_")[0]:
                kps_train.data.append(e)
            elif "test" in e.session.split("_")[0]:
                kps_test.data.append(e)
        if shuffle_data:
            kps_train, kps_test = shuffle_datasets(kps_train, kps_test)
    return kps_train, kps_test


def tests_lookup(files_list: list):
    """Create dict with all test. Used for faster testing."""
    out = dict()
    for file in files_list:
        if file != "all":
            _, kps_test = build_train_test(file)
        else:
            _, kps_test = build_from_all(files_list)
        out[file] = kps_test
    return out


def get_label(s):
    """Derive name of dataset based on file name"""
    return s.replace(".csv", "").replace("balanced", "").replace("_", "").replace("merged", "")


if __name__ == "__main__":
    iprint("Loading data...")

    if ALL_DATASETS:
        dataset_files = [DATA_ROOT + x for x in os.listdir(DATA_ROOT) if x.endswith(".csv")]
    else:
        dataset_files = DATASETS_LIST

    dataset_labels = {file_name: get_label(file_name) for file_name in dataset_files}
    tests = tests_lookup(dataset_files)

    preprocess_comb = ["raw", "fft", "mfcc"]
    wave_comb = ["thr", "th-", "t-r", "-hr", "t--", "-h-", "--r"]
    results = {data_training: {data_testing: {prep: {wave: None for wave in wave_comb} for prep in preprocess_comb} for data_testing in dataset_labels.values()} for data_training in dataset_labels.values()}
    for training_data in dataset_files:
        iprint(f"Starting to train on dataset {training_data}:")
        if training_data != "all":
            kps_train, kps_test = build_train_test(training_data)
        else:
            kps_train, kps_test = build_from_all(dataset_files)
        for preprocess in preprocess_comb:
            iprint(f"\twith preprocessing {preprocess}:")
            for wave in wave_comb:
                iprint(f"\t\t{wave}, against dataset:")
                fft_flag = False
                mfcc_flag = False
                if preprocess == "fft":
                    fft_flag = True
                if preprocess == "mfcc":
                    mfcc_flag = True
                subset = wave.replace("-", "")
                X_train, y_train = data_loader(kps_train,
                                               fft_flag=fft_flag,
                                               mfcc_flag=mfcc_flag,
                                               magnitude=True,
                                               subset=subset)

                # Check data shapes, just in case
                X_train_expected_shape = (N_KEYS * TRAIN_SIZE, len(subset) * SAMPLE_SIZE[preprocess])
                y_train_expected_shape = (N_KEYS * TRAIN_SIZE,)
                if training_data == "all":
                    X_train_expected_shape = ((len(DATASETS_LIST) - 1) * X_train_expected_shape[0], X_train_expected_shape[1])
                    y_train_expected_shape = ((len(DATASETS_LIST) - 1) * y_train_expected_shape[0],)
                assert X_train.shape == X_train_expected_shape, \
                       f"invalid X_train shape -- {X_train.shape} (expected {X_train_expected_shape})"
                assert y_train.shape == y_train_expected_shape, \
                       f"invalid y_train shape -- {X_train.shape} (expected {y_train_expected_shape})"

                # =====================
                # you will probably need to adjust this part to particular models
                # I assume that model have .fit(X, y) and .predict(X) methods
                # model = GaussianNB()
                # model = KNeighborsClassifier(1)
                # model = LogisticRegression(random_state=0, multi_class="ovr", max_iter=10000)
                model = svm.SVC(decision_function_shape="ovo")
                # model = xgboost_results_wrapper.XGBoostResultsWrapper()
                model = model.fit(X_train, y_train)
                # =====================
                for dataset in dataset_files:
                    subset = wave.replace("-", "")
                    X_test, y_test = data_loader(tests[dataset],
                                                 fft_flag=fft_flag,
                                                 mfcc_flag=mfcc_flag,
                                                 magnitude=True,
                                                 subset=subset)

                    # Check data shapes, just in case
                    X_test_expected_shape = (N_KEYS * TEST_SIZE, len(subset) * SAMPLE_SIZE[preprocess])
                    y_test_expected_shape = (N_KEYS * TEST_SIZE,)
                    if dataset == "all":
                        X_test_expected_shape = ((len(DATASETS_LIST) - 1) * X_test_expected_shape[0], X_test_expected_shape[1])
                        y_test_expected_shape = ((len(DATASETS_LIST) - 1) * y_test_expected_shape[0],)
                    assert X_test.shape == X_test_expected_shape, \
                           f"invalid X_test shape -- {X_test.shape} (expected {X_test_expected_shape})"
                    assert y_test.shape == y_test_expected_shape, \
                           f"invalid y_test shape -- {X_test.shape} (expected {y_test_expected_shape})"

                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    iprint(f"\t\t\t{dataset_labels[dataset]}:\t{acc}")
                    results[dataset_labels[training_data]][dataset_labels[dataset]][preprocess][wave] = acc
    # saving scheme: results[training_dataset][testing_dataset][preprocessing][peaks] = acc
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    iprint(f"Calculating results saved under {OUT_FILE}")
