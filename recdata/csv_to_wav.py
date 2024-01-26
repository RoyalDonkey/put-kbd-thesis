#!/usr/bin/env python3
"""Utilities for converting a number of .csv files in a format supported
by KeyPressSequence to corresponding .wav files (and taking a look at them
while you are at it)
"""
import wave
import sys
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from os.path import basename, dirname

sys.path.append("../pymodules")
from custom_read_csv import KeyPressSequence
from fft_tools import new_fft

PEAK_TYPES = ("touch", "hit", "release")
# 44100Hz / 4ms = 176.4, rounded down
# 4ms because of PEAK_WINDOW_MS defined in /recdata/wav_processing.py
SERIES_LEN = 176
# parameters based on parameters of our recording - source in
# /recdata/src/microphone.c
NUM_CHANNELS = 1
SAMPLE_RATE = 44100
SAMPLE_WIDTH_BYTES = 2

DEST_DIRECTORY = ""  # path to the directory where the new .wav files
# should be created
SHOW_DETAILED_PLOTS = False
SHOW_OVERVIEW_PLOTS = True
EXPORT_WAV = True


def kps_to_wav(kps: KeyPressSequence,
               out_filename: Optional[str] = "from_kps",
               frames_between_peaks: int = SAMPLE_RATE // 8,
               frames_between_keys: int = SAMPLE_RATE // 4) -> None:
    """Given a KeyPressSequence, write its data to a .wav file, inserting
    pauses as specified by function parameters
    """
    peak_pause = np.zeros(frames_between_peaks)
    key_pause = np.zeros(frames_between_keys)
    wav_object = wave.open(out_filename, "w")
    wav_object.setframerate(SAMPLE_RATE)
    wav_object.setnchannels(NUM_CHANNELS)
    wav_object.setsampwidth(SAMPLE_WIDTH_BYTES)
    # 3: touch, hit, release
    wav_object.setnframes(
        sum((
            3 * SERIES_LEN,
            2 * frames_between_peaks,
            frames_between_keys,
        )) * len(kps.data)
    )

    for kpd in kps.data:
        wav_object.writeframes(kpd.touch)
        wav_object.writeframes(peak_pause)
        wav_object.writeframes(kpd.hit)
        wav_object.writeframes(peak_pause)
        wav_object.writeframes(kpd.release)
        wav_object.writeframes(key_pause)

    wav_object.close()


def plot_all_peaks_of_type(
    kps: KeyPressSequence,
    peak_type: str,
    origin_filename: str,
    fft: bool = False,
) -> None:
    """Plots all peaks of a given type in the provided KeyPressSequence.
    Each series corresponding to a peak is labelled with its position within
    the kps
    """
    peak_type = peak_type.lower()
    assert peak_type in PEAK_TYPES, \
        f"Invalid peak type! Must be one of {PEAK_TYPES}"
    peak_list = None
    plot_color = None
    match peak_type:
        case "touch":
            peak_list = [kpd.touch for kpd in kps.data]
            plot_color = "red"
        case "hit":
            peak_list = [kpd.hit for kpd in kps.data]
            plot_color = "green"
        case "release":
            peak_list = [kpd.release for kpd in kps.data]
            plot_color = "blue"
        case _:
            assert False, f"unknown peak type: {peak_type}"
    peak_list = [new_fft(x, half=True) for x in peak_list] if fft else peak_list
    plt.title(f"{origin_filename}: {peak_type.upper()}")
    for i, peak in enumerate(peak_list):
        max_idx = np.argmax(peak)
        plt.annotate(i, xycoords='data',
                     xy=(max_idx, peak[max_idx]),
                     xytext=(1.5, 1.5), textcoords='offset points',
                     size="large", weight="bold")
        plt.plot(peak, color=plot_color, alpha=0.5)
    plt.show()


if __name__ == "__main__":
    filenames = sys.argv[1:]
    for filename in filenames:
        kps = KeyPressSequence(filename)
        if SHOW_DETAILED_PLOTS:
            plt.title(filename)
            for kpd in kps.data:
                # Uncomment for FFT plots
                # kpd.touch = new_fft(kpd.touch, half=True)
                # kpd.hit = new_fft(kpd.hit, half=True)
                # kpd.release = new_fft(kpd.release, half=True)
                plt.plot(kpd.touch, color="red")
                plt.plot(kpd.hit, color="green")
                plt.plot(kpd.release, color="blue")
                plt.show()
        if SHOW_OVERVIEW_PLOTS:
            for peak_type in PEAK_TYPES:
                plot_all_peaks_of_type(kps, peak_type, filename, fft=False)
        if EXPORT_WAV:
            key = dirname(filename).split('/')[-1]
            wav_filename = DEST_DIRECTORY + "peaks_" + key + "_" \
                + basename(filename).split('.')[0] + ".wav"
            kps_to_wav(kps, wav_filename)
