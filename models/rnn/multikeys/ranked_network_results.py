#!/usr/bin/env python3
import torch
import numpy as np
from typing import Literal
from os.path import basename
from rnn import RNNModel
import json

import sys
sys.path.append("../../../pymodules")
import KeyPressDataset as KeyPressDataset
from custom_read_csv import KeyPressSequence
from visualizations import prediction_heatmap


SERIES_LEN = 176         # 44100Hz / 4ms = 176.4, rounded down
NO_CLASSES = 43           # The number of output classes
VERBOSE = False

_config = {
    "subset": "thr",
    "batch_size": 8,
    "learning_rate": 0.002,
    "epochs": 50,
    "optimizer": "Adam",  # Name of optim class as string, e.g. "SGD" or "Adam"
    "momentum": 0.9,      # ignored if optimizer != "SGD"
    "preprocessing": "FFT",
}


def categorical_dicts(kps: KeyPressSequence
                      ) -> tuple[dict[str, int], dict[int, str]]:
    # copied here, to avoid using NO_CLASSES defined in rnn.py
    present_keys = set([kpd.key for kpd in kps.data])
    assert len(present_keys) == NO_CLASSES
    present_keys = list(present_keys)
    present_keys.sort(key=ord)
    key2int = dict(zip(present_keys, range(NO_CLASSES)))
    int2key = dict([(key2int[key], key) for key in key2int.keys()])
    return key2int, int2key


def filter_kps(kps: KeyPressSequence,
               key_whitelist: set | None = None,
               key_blacklist: set | None = None) -> KeyPressSequence:
    assert (key_whitelist or key_blacklist) and not (key_whitelist and key_blacklist), \
        "Must provide a whitelist XOR a blacklist"
    new_kps = KeyPressSequence()
    if key_whitelist:
        for kpd in kps.data:
            if kpd.key in key_whitelist:
                new_kps.data.append(kpd)
        del kps
        return new_kps
    if key_blacklist:
        indices_to_drop = set()
        for i, kpd in enumerate(kps.data):
            if kpd.key in key_blacklist:
                indices_to_drop.add(i)
        for i, kpd in enumerate(kps.data):
            if i in indices_to_drop:
                continue
            new_kps.data.append(kpd)
        del kps
        return new_kps


def normalize_kps(kps: KeyPressSequence) -> KeyPressSequence:
    for kpd in kps.data:
        kpd.touch = kpd.touch.astype(np.float32) / (2 ** 15)
        kpd.hit = kpd.hit.astype(np.float32) / (2 ** 15)
        kpd.release = kpd.release.astype(np.float32) / (2 ** 15)
    return kps


def top_predictions_dict(model_outputs: torch.Tensor,
                         categorical_labels: torch.Tensor,
                         key2int: dict[str, int],
                         int2key: dict[int, str],
                         ranks: int = 3) -> dict[str, np.ndarray[np.int32]]:
    """Dictionary structure:
        key: key from the test dataset which should have been predicted by the model
        value: np.ndarray representing how many times, given that the key
        represented by the corresponding dictionary key was the correct choice,
        it was actually predicted at a given rank
    Example:
    "a": np.array([8, 0, 1]) // in this example, ranks=3
    When 'a' was the correct key, the model correctly predicted is as the most likely key
    8 times, assigned it the second largest value 0 times, and third largest - once.
    Note that 'a' might have been the correct answer more than 8+0+1=9 times
    """
    out_dict = dict([(key, np.zeros(ranks, dtype=np.int32)) for key in key2int.keys()])
    for output, label in zip(model_outputs, categorical_labels):
        _, indices = torch.topk(output, ranks, sorted=True)
        rank = torch.where(indices == label, 1, 0)
        out_dict[int2key[int(label)]] += rank.numpy()
    return out_dict


def add_top_predictions_dicts(a: dict[str, np.ndarray[np.int32]],
                              b: dict[str, np.ndarray[np.int32]]
                              ) -> dict[str, np.ndarray[np.int32]]:
    """MODIFIES a
    """
    for k, v in b.items():
        a[k] += v
    return a


def export_top_predictions_dict(predictions_dict: dict[str, np.ndarray[np.int32]],
                                format: Literal["json"] | Literal["csv"],
                                filepath: str) -> None:
    match format:
        case "json":
            with open(f"{filepath}.json", "w") as f:
                listified = dict(zip(predictions_dict.keys(), [
                    nparr.tolist() for nparr in predictions_dict.values()
                ]))
                json.dump(listified, f, indent=4)
        case "csv":
            with open(f"{filepath}.csv", "w") as f:
                for k, v in predictions_dict.items():
                    joinable = list(map(str, v))
                    f.write(f"{k},{','.join(joinable)}\n")


def ranked_test(model: RNNModel, test_files: list[str], reported_ranks: int = 3
                ) -> tuple[float, dict[str, np.ndarray[np.int32]]]:
    """Tests a model's performance on a given data set.

    The testing dataset is formed by concatenating files on the `test_files` list.
    Returns accuracy and dictionary breaking down predictinos per character, inspired by "Keyboard Acoustic Emanations". See top_predictions_dict() for details
    """
    test_sequences = KeyPressSequence()
    for f in test_files:
        test_sequences += KeyPressSequence(f)

    # test_sequences = normalize_kps(test_sequences)
    test_dataset = KeyPressDataset.KeyPressDataset(kps=test_sequences,
                                                   preprocessing=_config["preprocessing"],
                                                   subset=_config["subset"])
    key2int, int2key = categorical_dicts(test_sequences)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset.data = [(t[0].flatten(), key2int[t[1]]) for t in test_dataset.data]

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=_config["batch_size"],
                                              shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        all_labels = []
        all_predictions = []
        prediction_dict = dict([(key, np.zeros(reported_ranks, dtype=np.int32)) for key in key2int.keys()])
        for data in test_loader:
            series, labels = data
            series = series.to(device)
            outputs, _ = model(series)
            prediction_dict = add_top_predictions_dicts(
                prediction_dict,
                top_predictions_dict(outputs.data, labels, key2int, int2key)
            )
            predictions = tuple(map(torch.argmax, outputs.data))
            total += len(predictions)
            correct += sum([x == y for x, y in zip(predictions, labels)])
            predictions = [int2key[int(p)] for p in predictions]
            labels = [int2key[int(l)] for l in labels]
            for prediction, label in zip(predictions, labels):
                all_predictions.append(prediction)
                all_labels.append(label)
        prediction_heatmap(all_predictions, all_labels,
                           sorted(list(key2int.keys())), f"RNN: {_config['epochs']} epochs {_config['subset']} {_config['preprocessing']}")
    print(f'Accuracy: {100 * correct / total:.2f} %')
    return correct / total, prediction_dict


if __name__ == "__main__":
    reported_ranks = 3
    model_path = sys.argv[1]  # "7744_RNN_balanced_thr_FFT_bsize=8_rlayers=1_dlayers=1_lr=002_e=50_opitm=Adam.pth"
    test_path = sys.argv[2:]  # "../../../data/balanced_test.csv"
    model = torch.load(model_path)
    _, top_dict = ranked_test(model, test_path, reported_ranks)
    export_filepath = f"top{reported_ranks}_" + basename(model_path).split(".")[0]
    export_top_predictions_dict(top_dict, "csv", export_filepath)
