#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchinfo
import torch.optim as optim
from typing import Optional
import time

import sys
sys.path.append("../../../pymodules")
import KeyPressDataset as KeyPressDataset
import custom_read_csv as crc


SERIES_LEN = 176         # 44100Hz / 4ms = 176.4, rounded down
NO_CLASSES = 43          # The number of output classes
VERBOSE = False           # Whether to print more information

_config = {
    "subset": "thr",
    "batch_size": 8,
    "learning_rate": 0.001,
    "epochs": 10,
    "optimizer": "Adam",  # Name of optim class as string, e.g. "SGD" or "Adam"
    "momentum": 0.9,      # ignored if optimizer != "SGD"
    "preprocessing": "MFCC"
}


class stacked(nn.Module):
    def __init__(self, pretrained_model: nn.Module):
        super().__init__()
        self.pretrained_model = pretrained_model
        if _config["preprocessing"] == "MFCC":
            self.dense_block = nn.Sequential(
                # model interactions within each series
                # nn.Linear(16, 16),  # 16 because of MFCC; usually: SERIES_LEN // deepvalbv3
                nn.Linear(1000, 16),
                nn.ReLU(),
            )
        # else:
        #     self.dense_block = nn.Sequential(
        #         # model interactions within each series
        #         nn.Linear(len(_config["subset"]), 1024),
        #         nn.ReLU(),
        #         nn.Linear(1024, 2048),
        #         nn.ReLU(),
        #     )
        self.out_activation = nn.Sequential(
            # aggregate each series to a single number
            # nn.Linear(16, 1), # deeplabv3
            nn.Linear(16, 16),
            nn.ReLU(),
            # reshape: [BATCH_SIZE, NO_SERIES, 1] -> [BATCH_SIZE, NO_SERIES]
            # nn.Flatten(),
            # combine the values representing all series into class probabilities
            # nn.Linear(21, NO_CLASSES), # deeplabv3
            nn.Linear(16, NO_CLASSES),
            nn.Softmax(dim=0),
        )

    def forward(self, x):
        # x = self.convolutional_block(x)
        # x = self.pretrained_model(x)['out'] // deeplabv3
        x = self.pretrained_model(x)
        x = self.dense_block(x)
        x = self.out_activation(x)
        return x


def categorical_dicts(kps: crc.KeyPressSequence
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


def train(train_files: list[str], pretrained_model: nn.Module) -> stacked:
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

    key2int, _ = categorical_dicts(train_sequences)
    train_dataset = KeyPressDataset.KeyPressDataset(
        kps=train_sequences,
        subset=_config["subset"],
        preprocessing=_config["preprocessing"]
    )
    train_dataset.data = [(t[0], key2int[t[1]])
                          for t in train_dataset.data]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_config["batch_size"],
                                               shuffle=True, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = stacked(pretrained_model)
    cnn = cnn.to(device)
    vprint(cnn)

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

    optimizer = optimizer_cls(cnn.parameters(), **optimizer_kwargs)
    epoch_times = []
    for epoch in range(_config["epochs"]):
        print(f"Starting epoch {epoch+1}/{_config['epochs']}")
        epoch_loss = 0
        epoch_start = time.time()
        hundred_sum = 0
        for i, data in enumerate(train_loader):
            iteration_start = time.time()
            if i % 100 == 99:
                print(f"EPOCH {epoch+1}: iterations {i-99}->{i} took {hundred_sum}")
                hundred_sum = 0
            inputs, labels = data
            # Convert labels to one-hot encoding
            labels = nn.functional.one_hot(labels.to(torch.int64), NO_CLASSES).to(torch.float)
            inputs = inputs.view(-1, len(_config["subset"]), 1, 16)
            labels = labels.to(device)
            inputs = inputs.to(device)
            # forward, backward, and then weight update
            optimizer.zero_grad()
            output = cnn(inputs)
            loss = loss_fn(output, labels)
            net_state = str(cnn.state_dict())
            loss.backward()
            epoch_loss += loss
            optimizer.step()
            if str(cnn.state_dict()) == net_state:
                vprint("Network not updating")
            hundred_sum += time.time() - iteration_start
        epoch_times.append((time.time() - epoch_start))

        print(f"Finished epoch {epoch+1} with loss: {epoch_loss}")
        torch.save(cnn, f"simple_stacked_NOTfrozen_resnet50_epoch{epoch+1}.pth")

    print(f"Avg epoch time: {np.mean(epoch_times)}; sum: {np.sum(epoch_times)}")
    print(epoch_times)
    print('Finished training stacked')
    return cnn


def test(model, test_files: list[str]) -> float:
    """Tests a model's performance on a given data set.

    The testing dataset is formed by concatenating files on the `test_files` list.
    Returns accuracy.
    """

    test_sequences = crc.KeyPressSequence()
    for f in test_files:
        test_sequences += crc.KeyPressSequence(f)
    key2int, int2key = categorical_dicts(test_sequences)
    test_dataset = KeyPressDataset.KeyPressDataset(kps=test_sequences,
                                                   preprocessing=_config["preprocessing"])
    test_dataset.data = [(t[0], key2int[t[1]]) for t in test_dataset.data]
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=_config["batch_size"],
                                              shuffle=False, num_workers=2)

    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for data in test_loader:
            series, labels = data
            series = series.view(-1, len(_config["subset"]), 1, 16)
            # series = series.transpose(1, 0)
            series = series.to(device)
            labels = labels.to(device)
            outputs = model(series)

            # predictions = tuple(map(lambda x: "k" if x[0] >= x[1] else "l", outputs.data))
            # total += len(predictions)
            # correct += sum([x == y for x, y in zip(predictions, labels)])

            # For multiclass outputs
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f} %')
    return correct / total


if __name__ == "__main__":
    model = resnet50(weights=ResNet50_Weights.DEFAULT,)

    # to train only the added layers, uncomment this
    # for param in model.parameters():
    #     param.requires_grad = False
    stacked_model = train(["data/balanced_train.csv"], model)
    torchinfo.summary(stacked_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stacked_model = stacked_model.to(device)
    test(stacked_model, ["data/balanced_test.csv"])
    torch.save(stacked_model, "simple_stacked_NOTfrozen_resnet50.pth")
