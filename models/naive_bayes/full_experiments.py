#!/usr/bin/env python3
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from typing import Optional
import numpy as np
import json

import sys

sys.path.append("../../pymodules")
import custom_read_csv as crc
from build_kps import build_kps
from fft_tools import new_fft, top_n, bottom_n, top_cutoff, bottom_cutoff


DATA_ROOT = "../../data/"
RELAXED_CUT = 0.90
VERBOSE = True
FRAMES_IN_PEAK = 176
OUT_FILE = "out.json"
SHUFFLE_DATA = False


def none(y: np.ndarray):
    """Helper function, just passes the given np array."""
    return y


def iprint(a: str, *args, **kwargs):
    """Helper function to print iff a VERBOSE flag is true"""
    if VERBOSE:
        print(a, *args, **kwargs)


def data_loader(
    kps: crc.KeyPressSequence,
    fft: bool = False,
    magnitude: bool = True,
    mode: Optional[dict] = None,
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
    # concatenate all columns into one with one list, depending on provided subset
    X = []
    for i, row in df.iterrows():
        t = []
        for name in features_names:
            t += row[name].tolist()
        X.append(t.copy())
        del t
    if fft:
        for i, row in enumerate(X):
            X[i] = new_fft(row, magnitude=magnitude)
            if mode:
                f = mode["func"]
                kwargs = mode["kwargs"]
                X[i] = f(X[i], **kwargs)
    y = df["key"].to_numpy()
    return np.array(X), y


def shuffle_datasets(ratio=1 / 11):
    RANDOM_SEED = 0
    # fraction of samples for each character used for testing
    # remaining - for training
    train_data = build_kps(DATA_ROOT, "train")
    test_data = build_kps(DATA_ROOT, "test")
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
    test_kps = crc.KeyPressSequence()
    train_kps = crc.KeyPressSequence()
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
    print(d)
    return train_kps, test_kps


if __name__ == "__main__":
    """Conduct experiments, testing all combinations of peaks,
    fft and cutout"""
    if SHUFFLE_DATA:
        kps_train, kps_test = shuffle_datasets()
    else:
        kps_train = build_kps(DATA_ROOT, "train")
        kps_test = build_kps(DATA_ROOT, "test")

    # create all possible combinations to test
    wave_comb = ["thr", "th-", "t-r", "-hr", "t--", "-h-", "--r"]
    all_mode_comb = dict()
    all_mode_comb["plain_fft"] = [{"func": none, "kwargs": {}}]
    all_mode_comb["top_n"] = [{"func": top_n, "kwargs": {"n": i}} for i in range(5, FRAMES_IN_PEAK, 5)]
    all_mode_comb["bottom_n"] = [{"func": bottom_n, "kwargs": {"n": i}} for i in range(5, FRAMES_IN_PEAK, 5)]
    all_mode_comb["top_cutoff"] = [{"func": top_cutoff, "kwargs": {"q": i * 0.01}} for i in range(1, 101)]
    all_mode_comb["bottom_cutoff"] = [{"func": bottom_cutoff, "kwargs": {"q": i * 0.01}} for i in range(1, 101)]
    modes_names = ["plain_fft", "top_n", "bottom_n", "top_cutoff", "bottom_cutoff"]
    results = {f"{name}-{mag=}": dict() for name in modes_names for mag in (True, False)}
    results["raw"] = dict()
    # iterat over all combinations
    for fft in [True, False]:
        if fft:
            for mag in [True, False]:
                for mode in modes_names:
                    iprint(f"Starting: {mode}, {mag=}")
                    mode_comb = all_mode_comb[mode]
                    for wave in wave_comb:
                        t = []
                        for comb in mode_comb:
                            # train calssyfier for particular combination of parameters
                            X_train, y_train = data_loader(kps_train, fft, mag, comb, wave.replace("-", ""))
                            X_test, y_test = data_loader(kps_test, fft, mag, comb, wave.replace("-", ""))
                            gnb = GaussianNB()
                            y_pred = gnb.fit(X_train, y_train).predict(X_test)
                            acc = accuracy_score(y_test, y_pred)
                            param = {"magnitude": mag,
                                     "transforming_function": comb["func"].__name__,
                                     "others": comb["kwargs"]}
                            t.append({"accuracy": acc, "parameters": param})
                            iprint(f"For {wave=}, {fft=}, {mag=}, {mode=} accuracy is equal to:  \t{acc:.3f}")
                        results[f"{mode}-{mag=}"][wave] = t
        # case when fft is turned off --> train on "raw" data
        else:
            for wave in wave_comb:
                t = []
                X_train, y_train = data_loader(kps_train,
                                               fft=None,
                                               magnitude=None,
                                               mode=None,
                                               subset=wave.replace("-", ""))
                X_test, y_test = data_loader(kps_test,
                                             fft=None,
                                             magnitude=None,
                                             mode=None,
                                             subset=wave.replace("-", ""))
                gnb = GaussianNB()
                y_pred = gnb.fit(X_train, y_train).predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                param = {"transforming_function": "none",
                         "others": dict()}
                t.append({"score": acc, "parameters": param})
                iprint(f"For {wave=}, mode=raw accuracy is equal to:  \t{acc:.3f}")
                results["raw"][wave] = t

    # save to .json file with such structure:
    # data[f"{name_of_function}-mag={bool_if_computed_with_magnitude}"][f"{t/h/r_combination}""],
    # which contains list with particular experiments and its accuracy:
    # {"accuracy": value,
    #  "parameters": {
    #     "magnitude": bool,
    #     "transforming_function": string,
    #     "other": {depending on transforming functino usually n or q etc.}}
    # }
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
