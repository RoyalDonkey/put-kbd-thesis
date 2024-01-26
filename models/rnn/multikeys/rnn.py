import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from typing import Optional
from os.path import basename

import sys
sys.path.append("../../../pymodules")
import KeyPressDataset as KeyPressDataset
from custom_read_csv import KeyPressSequence
from visualizations import prediction_heatmap  # noqa: F401

SERIES_LEN = 176         # 44100Hz / 4ms = 176.4, rounded down
NO_CLASSES = 43           # The number of output classes
VERBOSE = False

_config = {
    "subset": "thr",
    "batch_size": 8,
    "learning_rate": 0.002,
    "epochs": 50,
    "optimizer": "SGD",  # Name of optim class as string, e.g. "SGD" or "Adam"
    "momentum": 0.9,      # ignored if optimizer != "SGD"
    "preprocessing": "FFT",
}


def generate_export_filename(train_files: list[str],
                             recurrent_layers: int, dense_layers: int,
                             extension: str = ".pth") -> str:
    file_info = ""
    for file in train_files:
        file_info += basename(file).split("_")[0] + "_"
    fname = f"RNN_{file_info}{_config['subset']}_{_config['preprocessing']}_"
    fname += f"bsize={_config['batch_size']}_"
    fname += f"rlayers={recurrent_layers}_dlayers={dense_layers}_"
    fname += f"lr={str(_config['learning_rate'])[2:]}_e={_config['epochs']}_"
    fname += f"opitm={_config['optimizer']}"
    return fname + extension


def categorical_dicts(kps: KeyPressSequence
                      ) -> tuple[dict[str, int], dict[int, str]]:
    present_keys = set([kpd.key for kpd in kps.data])
    assert len(present_keys) == NO_CLASSES
    present_keys = list(present_keys)
    present_keys.sort(key=ord)
    key2int = dict(zip(present_keys, range(NO_CLASSES)))
    int2key = dict([(key2int[key], key) for key in key2int.keys()])
    return key2int, int2key


def vprint(*args, **kwargs) -> None:
    if not VERBOSE:
        return
    print(*args, **kwargs)


class RNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # RNN Layers
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layers
        self.dense_block = nn.Sequential(
            nn.Linear(hidden_dim, output_size),
        )
        self.device = device

    def forward(self, x):
        # Initialize hidden state for first input
        hidden = torch.zeros(self.n_layers, self.hidden_dim).to(self.device)

        # Pass input and hidden state into the model
        x, hidden = self.rnn(x, hidden)

        # Reshape the outputs to fit into the fully connected block
        x = x.contiguous().view(-1, self.hidden_dim)
        x = self.dense_block(x)
        return x, hidden


def seed(seed: int) -> None:
    """Sets the pytorch seed, use for deterministic results (default `None` = random)."""
    torch.manual_seed(seed)


def train(train_files: list[str], n_layers: int = 1) -> RNNModel:
    """Instantiates a new RNN and trains it on the given data.

    The training dataset is formed by concatenating files on the `train_files`
    list.
    """
    train_sequences = KeyPressSequence()
    for f in train_files:
        train_sequences += KeyPressSequence(f)

    key2int, _ = categorical_dicts(train_sequences)
    train_dataset = KeyPressDataset.KeyPressDataset(
        kps=train_sequences,
        subset=_config["subset"],
        preprocessing=_config["preprocessing"],
        mode="train"
    )
    train_dataset.data = [(t[0].flatten(), key2int[t[1]])
                          for t in train_dataset.data]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_config["batch_size"],
                                               shuffle=True, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _config["preprocessing"] == "MFCC":
        rnn = RNNModel(len(_config["subset"]) * 16, NO_CLASSES, len(_config["subset"]) * 16, n_layers, device)
    else:
        rnn = RNNModel(len(_config["subset"]) * SERIES_LEN, NO_CLASSES, len(_config["subset"]) * SERIES_LEN, n_layers, device)
    rnn = rnn.to(device)
    vprint(rnn)

    class verboseCrossEntropyLoss(nn.CrossEntropyLoss):
        """Performs the same calculations as pyTorch's CrossEntropyLoss,
        but allows us to see that is going on in the meantime - what value
        was computed for what data"""
        def __init__(self, weight: Optional[torch.Tensor] = None, size_average=None,
                     ignore_index: int = -100, reduce=None, reduction: str = 'mean',
                     label_smoothing: float = 0.0) -> None:
            super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)

        def forward(self, input, target):
            vprint(f"{input=}")
            vprint(f"{target=}")
            loss = super().forward(input, target)
            vprint(f"LOSS: {loss}")
            return loss

    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = verboseCrossEntropyLoss()

    # Construct optimizer kwargs
    optimizer_cls = optim.__getattribute__(_config["optimizer"])
    assert issubclass(optimizer_cls, optim.Optimizer), "optimizer is not a valid class name"
    optimizer_kwargs = {}
    optimizer_kwargs.update({"lr": _config["learning_rate"]})
    match optimizer_cls:
        case optim.SGD:
            if "momentum" in _config:
                optimizer_kwargs.update({"momentum": _config["momentum"]})

    optimizer = optimizer_cls(rnn.parameters(), **optimizer_kwargs)
    for epoch in range(_config["epochs"]):
        epoch_loss = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            # Convert labels to one-hot encoding
            labels = nn.functional.one_hot(labels.to(torch.int64), NO_CLASSES).to(torch.float)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # forward, backward, and then weight update
            optimizer.zero_grad()
            output, _ = rnn(inputs)  # ignore output of hidden state
            loss = loss_fn(output, labels)
            epoch_loss += loss
            net_state = str(rnn.state_dict())
            loss.backward()
            optimizer.step()
            if str(rnn.state_dict()) == net_state:
                vprint("Network not updating")
        print(f"Loss in epoch {epoch+1}/{_config['epochs']}: {loss}")
    print('Finished training RNNModel')
    # Export the trained model
    export_filename = generate_export_filename(train_files, n_layers, len(rnn.dense_block), ".pth")
    os.makedirs("rnn_models", exist_ok=True)
    torch.save(rnn, "rnn_models/" + export_filename)
    return rnn


def test(model: RNNModel, test_files: list[str]) -> float:
    """Tests a model's performance on a given data set.

    The testing dataset is formed by concatenating files on the `test_files` list.
    Returns accuracy.
    """
    test_sequences = KeyPressSequence()
    for f in test_files:
        test_sequences += KeyPressSequence(f)

    test_dataset = KeyPressDataset.KeyPressDataset(kps=test_sequences,
                                                   preprocessing=_config["preprocessing"],
                                                   subset=_config["subset"],
                                                   mode="test")
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
        for data in test_loader:
            series, labels = data
            series = series.to(device)
            outputs, _ = model(series)
            predictions = tuple(map(torch.argmax, outputs.data))
            total += len(predictions)
            correct += sum([x == y for x, y in zip(predictions, labels)])
            predictions = [int2key[int(p)] for p in predictions]
            labels = [int2key[int(l)] for l in labels]
            for prediction, label in zip(predictions, labels):
                all_predictions.append(prediction)
                all_labels.append(label)
        # prediction_heatmap(all_predictions, all_labels,
        #                   sorted(list(key2int.keys())), f"RNN: {_config['epochs']} epochs {_config['subset']} {_config['preprocessing']}")

    # print(f'Accuracy: {100 * correct / total:.2f} %')
    return float(correct / total)
