#!/usr/bin/env python3
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Optional
import json
import sys

sys.path.append("../../pymodules")
from custom_read_csv import KeyPressSequence
from fft_tools import new_fft
from mfcc_tools import mfcc

CSV_COLNAMES = ["key", "touch", "hit", "release"]
NUM_CLASSES = 43
NUM_BOOST_ROUND = 10  # Number of tree boosting rounds
RANDOM_SEED = 0


def configure() -> dict[str, Any]:
    # https://xgboost.readthedocs.io/en/latest/parameter.html

    # Global configuration
    print('-> Configuring XGBoost global parameters')
    xgb.set_config(verbosity=1, use_rmm=True)

    return {
        "booster"              : "gbtree",
        "learning_rate"        : 0.3,
        "min_split_loss"       : 0.0,
        "max_depth"            : 6,
        "min_child_weight"     : 1,
        "max_delta_step"       : 0.0,
        "subsample"            : 1.0,
        "sampling_method"      : "uniform",
        "reg_lambda"           : 1.0,
        "reg_alpha"            : 0.0,
        "tree_method"          : "auto",
        "process_type"         : "default",
        "grow_policy"          : "depthwise",
        "max_leaves"           : 0,
        "predictor"            : "auto",
        "num_parallel_tree"    : 1,
        # https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
        "objective"            : "multi:softmax",
        "num_class"            : NUM_CLASSES,
        "eval_metric"          : ["merror"],
        "seed"                 : RANDOM_SEED,
    }


def split_peak_columns(df: pd.DataFrame, skipped_columns: list[str] = ["key"]) -> pd.DataFrame:
    assert set(df.columns.to_list()) <= set(CSV_COLNAMES)
    new_columns = []
    split_columns = 0
    for colname in df.columns:
        if colname in skipped_columns:
            new_columns.append(colname)
            continue
        for i, _ in enumerate(df.iloc[0][colname]):
            new_columns.append(colname + '_' + str(i))
            split_columns += 1
    if split_columns == 0:
        raise ValueError("No columns to split")
    new_df = pd.DataFrame(columns=new_columns)
    new_values = []
    for _, row in df.iterrows():
        new_row = []
        for colname in df.columns:
            if colname in skipped_columns:
                new_row.append(row[colname])
            else:
                for i, elem in enumerate(row[colname]):
                    new_row.append(elem)
        new_values.append(new_row)
    new_df = pd.DataFrame(new_values, columns=new_columns)
    return new_df


def load_xgb_data(
        dataset_file: str = "../../data_balanced/main_balanced_merged.csv",
        preprocessing: str | None = None,
        shuffle_data: bool = False,
        subset: Optional[str] = None,
) -> tuple[xgb.DMatrix, xgb.DMatrix]:
    """Generate train and test sets for xgboost based on `dataset_file`.

    :param dataset_file: path to a merged csv file, defaults to `"../../data_balanced/main_balanced_merged.csv"`
    :type dataset_file: str, optional
    :param preprocessing: what type of operation should be applied to the data before it being transformed into a xgb.DMatrix.
        Currently supports: `None`, `"FFT"`, and `"MFCC"`. Defaults to None
    :type fft: str | None, optional
    :param subset: use only `"t"`ouch, `"h"`it, `"r"`elease, or any combination of data series (e.g. `"th"` for touch & release), defaults to `None`
    :type subset: str, optional
    :return: First element is the train xgb.DMatrix, second - test
    :rtype: tuple[xgb.DMatrix, xgb.DMatrix]
    """
    assert preprocessing in (None, "FFT", "MFCC")

    main_kps = KeyPressSequence(dataset_file)
    train_data = KeyPressSequence()
    test_data = KeyPressSequence()
    if not shuffle_data:
        for kpd in main_kps.data:
            if kpd.session[:4] == "test":
                test_data.data.append(kpd)
            else:
                train_data.data.append(kpd)
    if shuffle_data:
        num_test_examples = int(len(main_kps) / 11)
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(main_kps.data)
        test_data.data = main_kps.data[:num_test_examples]
        train_data.data = main_kps.data[num_test_examples:]

    train_df = train_data.to_pandas()
    test_df = test_data.to_pandas()
    if subset is not None:
        assert set(subset) <= set("thr"), "illegal subset value"
        assert len(subset) != 0, "subset cannot be empty"
        col_names = ["key"] + [{"t": "touch", "h": "hit", "r": "release"}[c] for c in subset]
        train_df = train_df[col_names]
        test_df = test_df[col_names]

    if preprocessing == "FFT":
        for df in (train_df, test_df):
            for colname in df.columns:
                if colname == "key":
                    continue
                df[colname] = df[colname].map(new_fft)
    elif preprocessing == "MFCC":
        for df in (train_df, test_df):
            for colname in df.columns:
                if colname == "key":
                    continue
                # [0] ensures correct dimensionality
                df[colname] = df[colname].map(lambda x: mfcc(x)[0])

    train_df = split_peak_columns(train_df)
    test_df = split_peak_columns(test_df)
    assert set(train_df["key"]) == set(test_df["key"])
    present_keys = list(set(train_df["key"]))
    assert len(present_keys) == NUM_CLASSES, \
        f"Suggested classes: {NUM_CLASSES}, keys in dataset: {len(present_keys)}"
    present_keys.sort(key=lambda x: ord(x))
    label_dict = dict(zip(present_keys, range(NUM_CLASSES)))
    train_peaks = train_df.iloc[:, 1:]
    train_keys = pd.DataFrame([label_dict[key] for key in train_df["key"]], columns=["key"])
    train_keys = train_keys.astype("category")
    test_peaks = test_df.iloc[:, 1:]
    test_keys = pd.DataFrame([label_dict[key] for key in test_df["key"]], columns=["key"])
    test_keys = test_keys.astype("category")
    xgb_train = xgb.DMatrix(train_peaks, train_keys["key"], enable_categorical=True)
    xgb_test = xgb.DMatrix(test_peaks, test_keys["key"],  enable_categorical=True)

    return xgb_train, xgb_test


def evaluate(bst: xgb.Booster, evals_result: dict[Any, Any], deval: xgb.DMatrix) -> float:
    acc = 1. - evals_result['deval']['merror'][-1]
    return acc


def plot(bst: xgb.Booster) -> None:
    print('-> Generating plots')
    fig, ax = plt.subplots(figsize=(15, 15))
    xgb.plot_tree(bst, ax=ax, num_trees=0)
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 15))
    xgb.plot_importance(bst, ax=ax)
    plt.show()


def train_model(dtrain: xgb.DMatrix, evallist: list, evals_result: dict[Any, Any], params: Optional[dict[str, Any]] = None) -> xgb.Booster:
    if params is None:
        params = configure()
    bst = xgb.train(params, dtrain, NUM_BOOST_ROUND, evals=evallist, evals_result=evals_result)
    return bst


def export_model(bst: xgb.Booster, evals_result: dict[Any, Any], model_fname: str = "xgboost.model", evals_fname: str = "evals_result.json") -> None:
    with open(evals_fname, "w") as f:
        json.dump(evals_result, f)
    bst.save_model(model_fname)


def load_model(model_fname: str = "xgboost.model", evals_fname: str = "evals_result.json") -> tuple[xgb.Booster, dict[Any, Any]]:
    bst = xgb.Booster()
    bst.load_model(model_fname)
    with open(evals_fname) as f:
        evals_result = json.load(f)
    return bst, evals_result


if __name__ == "__main__":
    dtrain, deval = load_xgb_data(preprocessing="MFCC")
    params = configure() | {
        "min_split_loss"    : 0.2,
        "min_child_weight"  : 0.2,
    }

    bst: Optional[xgb.Booster] = None
    evals_result: Optional[dict[Any, Any]] = None

    # Load data (comment out to train & export from scratch)
    bst, evals_result = load_model()

    if bst is None or evals_result is None:
        evallist = [(deval, 'deval')]
        evals_result = {}
        bst = train_model(dtrain, evallist, evals_result, params)
        export_model(
            bst,
            evals_result,
            "xgboost.model",
            "evals_result.json"
        )

    plot(bst)
    importance_dict = bst.get_score(importance_type='gain')
    importance_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
    print(importance_dict)
    print("Accuracy:", evaluate(bst, evals_result, deval))
