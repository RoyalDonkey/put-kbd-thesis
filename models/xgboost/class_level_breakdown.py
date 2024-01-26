#!/usr/bin/env python3
import xgboost as xgb
import pandas as pd
from typing import Any, Optional
import json
from xgboost_classifier import split_peak_columns, NUM_CLASSES
from sklearn.metrics import accuracy_score, confusion_matrix

import sys
sys.path.append("../../pymodules")
from custom_read_csv import KeyPressSequence
from fft_tools import new_fft
from build_kps import build_kps


def load_model(model_fname: str = "xgboost.model", evals_fname: str = "evals_result.json") -> tuple[xgb.Booster, dict[Any, Any]]:
    bst = xgb.Booster()
    bst.load_model(model_fname)
    with open(evals_fname) as f:
        evals_result = json.load(f)
    return bst, evals_result


def load_xgb_test_data(
        dataset_root: str = "../../data/",
        fft: bool = False,
        subset: Optional[str] = None,
) -> tuple[pd.DataFrame, xgb.DMatrix, dict[int, str]]:
    """Generate test data for xgboost based on files under the `dataset_root`

    :param dataset_root: root of dataset directory, defaults to `"../../data/"`
    :type dataset_root: str, optional
    :param fft: whether to use FFT'd data instead of raw waveforms, defaults to `False`
    :type fft: bool, optional
    :param subset: use only `"t"`ouch, `"h"`it, `"r"`elease, or any combination of data series (e.g. `"th"` for touch & release), defaults to `None`
    :type subset: str, optional
    :return: First element is a pd.DataFrame with the test labels, used for easy
    presentability, the second - the xgb.DMatrix used by the model, third -
    a dictionary mapping labels (numbers, which xgb uses) to characters
    :rtype: tuple[pd.DataFrame, xgb.DMatrix, dict]
    """
    test_data = KeyPressSequence()

    test_data = build_kps(dataset_root, "test")

    test_df = test_data.to_pandas()
    if subset is not None:
        assert set(subset) <= set("thr"), "illegal subset value"
        assert len(subset) != 0, "subset cannot be empty"
        col_names = ["key"] + [{"t": "touch", "h": "hit", "r": "release"}[c] for c in subset]
        test_df = test_df[col_names]

    if fft:
        for colname in test_df.columns:
            if colname == "key":
                continue
            test_df[colname] = test_df[colname].map(new_fft)

    test_df = split_peak_columns(test_df)
    present_keys = list(set(test_df["key"]))
    assert len(present_keys) == NUM_CLASSES, \
        f"Suggested classes: {NUM_CLASSES}, keys in dataset: {len(present_keys)}"
    present_keys.sort(key=lambda x: ord(x))
    label_dict = dict(zip(present_keys, range(NUM_CLASSES)))
    test_peaks = test_df.iloc[:, 1:]
    test_keys = pd.DataFrame([label_dict[key] for key in test_df["key"]], columns=["key"])
    test_keys = test_keys.astype("category")
    xgb_test = xgb.DMatrix(test_peaks, test_keys["key"],  enable_categorical=True)
    ret_df = pd.DataFrame(test_df.iloc[:, 0]).join(test_keys.iloc[:, :], rsuffix="_code")
    reverse_dict = dict(zip(label_dict.values(), label_dict.keys()))
    return ret_df, xgb_test, reverse_dict


if __name__ == "__main__":
    test_chars, deval, decoding_dict = load_xgb_test_data()
    bst, evals_result = load_model()
    ret = bst.predict(deval).astype(int)
    predicted_keys = [decoding_dict[x] for x in ret]
    print("Overall accuracy:", accuracy_score(test_chars["key"], predicted_keys))
    present_chars = sorted(list(set(test_chars["key"])))
    test_confusion_matrix = confusion_matrix(test_chars["key"], predicted_keys, labels=present_chars)
    c_acc = test_confusion_matrix.diagonal() / test_confusion_matrix.sum(axis=1)
    print("Class level accuracies:")
    for char, acc in zip(present_chars, c_acc):
        print(char, acc)
