"""
This module provides the `KeyPressDataset` class, which wraps a
`KeyPressSequence` object in a `torch.utils.data.Dataset` object, making it
suitable for use in neural networks.
"""
from custom_read_csv import KeyPressSequence
import numpy as np
from torch.utils.data import Dataset
from torch import from_numpy
from sklearn.preprocessing import normalize
from typing import Optional
from fft_tools import new_fft
from mfcc_tools import mfcc

SAMPLE_SIZE = {
    None: 176,
    "FFT": 176,
    "MFCC": 16,
}


class KeyPressDataset(Dataset):
    def __init__(
            self,
            data_fpath: Optional[str] = None,
            kps: Optional[KeyPressSequence] = None,
            subset: Optional[str] = None,  # e.g. "th" for only touch and hit
            preprocessing: Optional[str] = None,
            mode: Optional[str] = None,
    ):
        assert data_fpath is not None or kps is not None
        assert data_fpath is None or kps is None
        assert preprocessing in (None, "FFT", "MFCC")
        assert mode in (None, "test", "train")
        if data_fpath is not None:
            kps = KeyPressSequence(data_fpath)
        if subset is not None:
            assert set(subset) <= set("thr"), "illegal subset value"
        else:
            subset = "thr"
        assert isinstance(kps, KeyPressSequence)

        if mode is not None:
            for i in sorted(range(len(kps.data)), reverse=True):
                try:
                    if mode == "test":
                        if "train" in kps.data[i].session.split("_")[0]:
                            del kps.data[i]
                    elif mode == "train":
                        if "test" in kps.data[i].session.split("_")[0]:
                            del kps.data[i]
                except AttributeError:
                    print("Session was not found in kpd, cannot decide whether it is a train or test instance")
        labels = [x.key for x in kps]  # type: ignore
        arr_touch = np.array([x.touch for x in kps]).astype(float)  # type: ignore
        arr_hit = np.array([x.hit for x in kps]).astype(float)  # type: ignore
        arr_release = np.array([x.release for x in kps]).astype(float)  # type: ignore
        if preprocessing == "FFT":
            for i in range(len(arr_touch)):
                arr_touch[i] = new_fft(arr_touch[i])
                arr_hit[i] = new_fft(arr_hit[i])
                arr_release[i] = new_fft(arr_release[i])
        elif preprocessing == "MFCC":
            mfcc_touch = np.zeros((arr_touch.shape[0], SAMPLE_SIZE["MFCC"]))
            mfcc_hit = np.zeros((arr_hit.shape[0], SAMPLE_SIZE["MFCC"]))
            mfcc_release = np.zeros((arr_release.shape[0], SAMPLE_SIZE["MFCC"]))

            for i in range(len(arr_touch)):
                mfcc_touch[i] = mfcc(arr_touch[i])
                mfcc_hit[i] = mfcc(arr_hit[i])
                mfcc_release[i] = mfcc(arr_release[i])
            arr_touch = mfcc_touch
            arr_hit = mfcc_hit
            arr_release = mfcc_release
        arr_touch = normalize(arr_touch, norm="max")
        arr_hit = normalize(arr_hit, norm="max")
        arr_release = normalize(arr_release, norm="max")

        expected_shape = (len(kps), SAMPLE_SIZE[preprocessing])
        assert arr_touch.shape == expected_shape, \
               f"Invalid arr_touch shape: {arr_touch.shape} (expected {expected_shape})"
        assert arr_hit.shape == expected_shape, \
               f"Invalid arr_hit shape: {arr_hit.shape} (expected {expected_shape})"
        assert arr_release.shape == expected_shape, \
               f"Invalid arr_release len: {arr_release.shape} (expected {expected_shape})"
        assert len(labels) == len(arr_touch) == len(arr_hit) == len(arr_release)

        self.data = []
        for label, t, h, r in zip(labels, arr_touch, arr_hit, arr_release):
            arr = []
            if "t" in subset:
                arr.append(t)
            if "h" in subset:
                arr.append(h)
            if "r" in subset:
                arr.append(r)
            assert len(arr) == len(subset), f"Invalid arr len: {len(arr)} (expected {len(subset)})"
            tensor = from_numpy(np.array(arr))
            self.data.append((tensor.float(), label))

        assert len(self.data) == len(kps), \
               f"Invalid self.data length: {len(self.data)} (expected {len(kps)})"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
