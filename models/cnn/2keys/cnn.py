"""
This module implements CNNs for supervised classification of two keyboard keys
based on their sound. For all implementations, the input and output formats are
the same:
INPUT:  array of shape (3, 176)
OUTPUT: predicted character

Adapted from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Any

import sys
sys.path.append("../../../pymodules")
import KeyPressDataset as KeyPressDataset
import custom_read_csv as crc

SERIES_LEN = 176         # 44100Hz / 4ms = 176.4, rounded down
NO_CLASSES = 2           # The number of output classes
VERBOSE = True           # Whether to print more information

_config = {
    "subset": "thr",
    "batch_size": 8,
    "learning_rate": 0.001,
    "epochs": 20,
    "optimizer": "Adam",  # Name of optim class as string, e.g. "SGD" or "Adam"
    "momentum": 0.9,      # ignored if optimizer != "SGD"
}


def vprint(*args, **kwargs) -> None:
    if not VERBOSE:
        return
    print(*args, **kwargs)


def configure(config: dict[str, Any]) -> None:
    _config.update(config)


class CnnBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolutional_block = nn.Sequential(
            nn.Conv1d(len(_config["subset"]), 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.dense_block = nn.Sequential(
            # model interactions within each series
            nn.Linear(SERIES_LEN, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.out_activation = nn.Sequential(
            # aggregate each series to a single number
            nn.Linear(2048, 1),
            nn.ReLU(),
            # reshape: [BATCH_SIZE, NO_SERIES, 1] -> [BATCH_SIZE, NO_SERIES]
            nn.Flatten(),
            # combine the values representing all series into class probabilities
            nn.Linear(256, NO_CLASSES),
            nn.Softmax(dim=0),
        )

    def forward(self, x):
        x = self.convolutional_block(x)
        x = self.dense_block(x)
        x = self.out_activation(x)
        return x


def seed(seed: int) -> None:
    """Sets the pytorch seed, use for deterministic results (default `None` = random)."""
    torch.manual_seed(seed)


def train(train_files: list[str]) -> CnnBasic:
    """Instantiates a new CNN and trains it on the given data.

    The training dataset is formed by concatenating files on the `train_files`
    list.
    """
    train_sequences = crc.KeyPressSequence()
    for f in train_files:
        train_sequences += crc.KeyPressSequence(f)
    for kps in train_sequences.data:
        kps.touch = kps.touch.astype(np.float32) / (2 ** 15)
        kps.hit = kps.hit.astype(np.float32) / (2 ** 15)
        kps.release = kps.release.astype(np.float32) / (2 ** 15)
        if kps.key == "k":
            kps.key = np.float32(0)
        if kps.key == "l":
            kps.key = np.float32(1)

    train_dataset = KeyPressDataset.KeyPressDataset(
        kps=train_sequences,
        subset=_config["subset"],
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_config["batch_size"],
                                               shuffle=True, num_workers=2)

    cnn = CnnBasic()
    vprint(cnn)

    class my_class(nn.CrossEntropyLoss):
        def __init__(self, weight: Optional[torch.Tensor] = None, size_average=None, ignore_index: int = -100,
                     reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
            super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)

        def forward(self, input, target):
            vprint(f"{input=}")
            vprint(f"{target=}")
            loss = super().forward(input, target)
            vprint(f"LOSS: {loss}")
            # vprint("+-----------------+")
            return loss

    # loss_fn = nn.BCELoss()
    loss_fn = my_class()

    # Construct optimizer kwargs
    optimizer_cls = optim.__getattribute__(_config["optimizer"])
    assert issubclass(optimizer_cls, optim.Optimizer), "optimizer is not a valid class name"
    optimizer_kwargs = {}
    optimizer_kwargs.update({"lr": _config["learning_rate"]})
    match optimizer_cls:
        case optim.SGD:
            if "momentum" in _config:
                optimizer_kwargs.update({"momentum": _config["momentum"]})

    optimizer = optimizer_cls(cnn.parameters(), **optimizer_kwargs)
    for epoch in range(_config["epochs"]):
        for i, data in enumerate(train_loader):
            inputs, labels = data

            # Convert labels to one-hot encoding
            labels = nn.functional.one_hot(labels.to(torch.int64), NO_CLASSES).to(torch.float)

            # forward, backward, and then weight update
            optimizer.zero_grad()
            output = cnn(inputs)
            loss = loss_fn(output, labels)
            net_state = str(cnn.state_dict())
            loss.backward()
            optimizer.step()
            if str(cnn.state_dict()) == net_state:
                vprint("Network not updating")
    vprint('Finished training CnnBasic')
    return cnn


def test(model: CnnBasic, test_files: list[str]) -> float:
    """Tests a model's performance on a given data set.

    The testing dataset is formed by concatenating files on the `test_files` list.
    Returns accuracy.
    """
    test_sequences = crc.KeyPressSequence()
    for f in test_files:
        test_sequences += crc.KeyPressSequence(f)
    test_dataset = KeyPressDataset.KeyPressDataset(kps=test_sequences)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=_config["batch_size"],
                                              shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            predictions = tuple(map(lambda x: "k" if x[0] >= x[1] else "l", outputs.data))
            total += len(predictions)
            correct += sum([x == y for x, y in zip(predictions, labels)])

            # For multiclass outputs
            # _, predicted = torch.max(outputs.data, 1)
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

    vprint(f'Accuracy: {100 * correct / total:.2f} %')
    return correct / total
