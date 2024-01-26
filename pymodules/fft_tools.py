#!/usr/bin/env python3
"""
Small module with useful methods to transform frequencies to their fft forms,
get only top-n frequencies, etc.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
from typing import Optional, Callable

sys.path.append("../../pymodules")
from build_kps import build_kps  # noqa: E402

DEBUG = False
DATA_ROOT = "../data/"
SAMPLING_RATE = 44100
FRAMES_IN_PEAK = 176
FREQ = 1 / SAMPLING_RATE
MEASUREMENT_PERIOD = FREQ * FRAMES_IN_PEAK


def new_fft(y: np.ndarray,
            freq: float = FREQ,
            y_only: bool = True,  # if true returns only real part of fft
            magnitude: bool = True,
            half: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Correct way to calculate fft. Please remember to specify frequency of signal."""
    postprocess = (lambda x: x) if not half else (lambda x: x[:round(len(x) / 2)])
    y_fft = scipy.fft.fft(y)
    # y_fft = y_fft.real
    if magnitude:
        y_fft = np.abs(y_fft)
    if y_only:
        return postprocess(y_fft.real)
    else:
        x_fft = scipy.fft.fftshift(scipy.fft.fftfreq(y_fft.size, freq))
        return postprocess(x_fft), postprocess(y_fft)


def bottom_n(y: np.ndarray, n: int = 75) -> np.ndarray:
    """Zeroes out all values greater than the nth largest value in y."""

    assert n > 0, "n should be greater than zero and integer"
    assert n < len(y), "n should be smaller than size of list and integer"
    thresh = np.argsort(y, )[:n]
    mask = np.ones(y.size, dtype=bool)
    mask[thresh] = False
    y[mask] = 0
    if DEBUG:
        print(f"Values preserved in bottom {n}:", np.count_nonzero(y))
    return y


def top_n(y: np.ndarray, n: int = 7) -> np.ndarray:
    """Zeroes out all values lower than the nth largest value in y."""
    assert n > 0, "n should be greater than zero and integer"
    assert n < len(y), "n should be smaller than size of list and integer"
    thresh = np.argsort(y, )[len(y) - n:]
    mask = np.ones(y.size, dtype=bool)
    mask[thresh] = False
    y[mask] = 0
    if DEBUG:
        print(f"Values preserved in top {n}:", np.count_nonzero(y))
    return y


def top_cutoff(y: np.ndarray, q: float = 0.8) -> np.ndarray:
    """Zeroes out all values of normalized y lower than q."""
    mx = np.max(y)
    mn = np.min(y)
    y[y <= (q * (mx - mn) + mn)] = 0
    if DEBUG:
        print("Values preserved in top cutoff:", np.count_nonzero(y))
    return y


def bottom_cutoff(y: np.ndarray, q: float = 0.2) -> np.ndarray:
    """Zeroes out all values of normalized y higher than q."""
    mx = np.max(y)
    mn = np.min(y)
    y[y >= (q * (mx - mn) + mn)] = 0
    if DEBUG:
        print("Values preserved in bottom cutoff:", np.count_nonzero(y))
    return y


def test_wave(y_only: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Very good function to test fft and plot it.

    Taken from wiki: https://en.wikipedia.org/wiki/Fourier_transform#Example"""
    x = np.arange(-2, 2, 1 / 100)
    y = np.cos(2 * np.pi * (3 * x)) * np.exp(-np.pi * x * x)
    if y_only:
        return y
    else:
        return x, y


def dumb_wave(density: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Signal from tutorial about fft density - number of samples to take."""
    def f(x):
        return 5 + 2 * np.cos(2 * np.pi * x - np.pi / 2) + 3 * np.cos(4 * np.pi * x)
    _x = np.linspace(0, 2.5, density)
    return _x, f(_x)


def test_real_wave() -> tuple[np.ndarray, np.ndarray]:
    kps = build_kps(DATA_ROOT, "train")
    x = np.arange(0, MEASUREMENT_PERIOD, FREQ)
    y = kps[2].touch
    return x, y


def prove_fft(x_original: np.ndarray,
              y_original: np.ndarray,
              y_fft: np.ndarray,
              sidewise: bool = True) -> None:
    """Inverse Fourier transform from scratch, allows to show
    consecutive approximations of the original function via y_fft
    sidewise - show original signal and approximated in two different plots."""
    N = len(x_original)
    estimated_y_original = [[0 for i in range(len(y_fft))] for j in range(N)]
    for i, x in enumerate(x_original):
        for j, fft_value in enumerate(y_fft):
            estimated_y_original[j][i] = fft_value * np.exp(complex(0, 1) * (2 * np.pi / N) * i * j) / N
    estimated_y_original = np.cumsum(estimated_y_original, axis=0).real  # type: ignore
    if sidewise:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].plot(x_original, y_original, linewidth=2, color="black")
        ax[0].set_title("Original signal")
        for i, estimation in enumerate(estimated_y_original):
            ax[1].plot(estimation, linewidth=0.2, color="green")
            ax[1].set_title("Cumulative approximation")
    else:
        plt.plot(x_original, y_original, linewidth=2, color="black")
        for estimation in estimated_y_original:
            plt.plot(estimation, linewidth=0.1, color="green")
        plt.title("Original signal (black) and its approximation")
    if DEBUG:
        print("Sum of errors is equal to:", np.sum(np.abs(y_original - estimated_y_original[-1])))
    plt.show()


def simple_plot(x, y, title=None):
    plt.plot(x, y.real)
    if title:
        plt.title(title)
    plt.show()


def compare_plot(x: np.ndarray,
                 y: np.ndarray,
                 f: Callable,
                 kwargs: Optional[dict] = None,
                 title: Optional[str] = None) -> None:
    """Helper method, mainly used to compare values of Fourier transform after cutoff
    or getting top values."""
    plt.plot(x, y, color="red", alpha=0.3)
    if kwargs:
        plt.plot(x, f(y, **kwargs), color="green")
    else:
        plt.plot(x, f(y), color="green")

    if title:
        plt.title(title)
    plt.show()


if __name__ == "__main__":
    DEBUG = True
    # x, y = test_wave()
    # x, y = dumb_wave(10)
    # x, y = dumb_wave(100)
    x, y = test_real_wave()
    x_fft, y_fft = new_fft(y, FREQ, y_only=False, magnitude=False)

    prove_fft(x, y, y_fft)
    simple_plot(x_fft, y_fft, "Data after Fourier transform")
    compare_plot(x_fft, np.abs(y_fft), top_cutoff, {"q": 0.8}, title="Top cutoff")
    compare_plot(x_fft, np.abs(y_fft), bottom_cutoff, {"q": 0.2}, title="Bottom cutoff")
    compare_plot(x_fft, np.abs(y_fft), top_n, {"n": 20}, title="Top n")
    compare_plot(x_fft, np.abs(y_fft), bottom_n, {"n": 88}, title="Bottom n")
