#!/usr/bin/env python3

import struct, sys, wave
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from typing import Optional
from os.path import dirname, basename
from pathlib import Path


def read_keys_file(filename: str) -> tuple:
    keys_file = open(f"{filename}.keys")
    lines = keys_file.readlines()
    keys_file.close()
    times = []
    keys = []
    for i, line in enumerate(lines):
        if line[0] == '#':
            print(f'skipped comment on line {i}:', line.strip('\n'))
            continue
        time, key = line[:line.find(' ')], line.strip('\n')[-1]
        times.append(float(time))
        keys.append(key)
    return times, keys


def read_frame_values_from_wav(wav_file_object: wave.Wave_read) -> np.ndarray:
    frames = []
    for _ in range(wav_file_object.getnframes()):
        wave_data = wav_file_object.readframes(1)
        data = struct.unpack("<h", wave_data)[0]
        frames.append(data)
    return np.array(frames)


# interval of recording before and after keypress was registered to search
MAIN_WINDOW_MS = (-10, 90)
# once a peak is found, extract this much of the recording from the section surrounding it
PEAK_WINDOW_MS = (-2, 2)  # original: -2, 2
global OVERLAPS
OVERLAPS = 0
global KEY_COUNT
KEY_COUNT = 0


def usage():
    print('Usage:')
    print('    wav_processing.py [-f, -p, -h] FILE...')
    print('-h: print this usage info')
    print('-f: force overwriting output files')
    print('-p: show plots for each processed file')
    print("-n: creates a new dataset structure to store the new csvs")
    print("the root of this structure MUST BE SPECIFIED WITHIN the code")
    print()
    print('Each FILE path must either be extensionless or end with ".wav".')
    print('It is assumed that a corresponding ".keys" file will exist.')
    print('Output is saved under the same dir and name as each input, with extension ".csv".')


def extract_keypress_frames(time: float, frames: npt.NDArray,
                            sampling_rate: int = 44100) -> tuple[npt.NDArray, ...]:
    """
    Returned windows are sorted chronologically
    """
    global KEY_COUNT
    KEY_COUNT += 1
    press_frame_idx = int(time * sampling_rate)
    # wontdo: refactor for this to be done once per run, not every window
    # unnecessary optimization
    init_frame_idx = int(press_frame_idx + MAIN_WINDOW_MS[0] * sampling_rate / 1000)
    ending_frame_idx = int(init_frame_idx + MAIN_WINDOW_MS[1] * sampling_rate / 1000)
    peak_window_frames = tuple([int(x * sampling_rate / 1000) for x in PEAK_WINDOW_MS])
    # ===

    window_frames = frames[init_frame_idx:ending_frame_idx + 1]
    frame_idx_pairs = list(zip(range(init_frame_idx, ending_frame_idx),
                               window_frames.tolist()))
    frame_idx_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    peaks = [(-1, None) for _ in range(3)]
    peaks[0] = frame_idx_pairs[0]
    found_peaks = 1

    # keep the "overlapping window" conundrum in mind
    def is_idx_outside_window(midpoint: int, window: tuple, idx: int) -> bool:
        return not midpoint + window[0] < idx < midpoint + window[1]

    def is_there_overlap(earlier_mid: int, later_mid: int) -> bool:
        # note how the range defining keypress_frames is computed
        # that's what the right side of this inequality must correspond
        # in value to - the number of frames within a single peak
        return later_mid - earlier_mid >= sum(map(abs, peak_window_frames))
    for idx, val in frame_idx_pairs[1:]:
        new_peak_found = True
        # check that idx is outside of all windows found so far
        for i in range(found_peaks):
            if not is_idx_outside_window(peaks[i][0], peak_window_frames, idx):
                new_peak_found = False
                break
        if not new_peak_found:
            continue
        peaks[found_peaks] = idx, val
        found_peaks += 1
        if found_peaks == 3:
            break

    peaks.sort(key=lambda x: x[0])
    # this line is the reason all peaks have 176 elements, not 177
    # python slices are [), so we get frames from [x-88, x+88) - 176 total
    # (88 before the peak frame, 1 peak frame, and 87 afterwards)
    keypress_frames = tuple([frames[peak[0] + peak_window_frames[0]:
                                    peak[0] + peak_window_frames[1]]
                             for peak in peaks])
    for i in range(len(peaks) - 1):
        if is_there_overlap(peaks[i][0], peaks[i + 1][0]):
            global OVERLAPS
            OVERLAPS += 1
    return keypress_frames


def write_to_csv(keys: list[str],
                 peak_frames: list[tuple[npt.NDArray, ...]],
                 input_filename: str,
                 data_path: Optional[str] = None,
                 force: bool = False) -> None:
    # If no data_path is given, reuse input file's directory
    output_dirname = data_path if data_path is not None else dirname(input_filename)
    output_filename = f"{output_dirname}/{basename(input_filename)}.csv"
    # Check if the file already exists
    if not force:
        try:
            file = open(output_filename, "r")
            print(f"output file '{output_filename}' already exists. pass '-f' to force")
            sys.exit(1)
        except FileNotFoundError:
            pass
    file = open(output_filename, "w+")
    file.write("key,touch,hit,release\n")
    touch, hit, release = zip(*peak_frames)
    for key, t, h, r in zip(keys, touch, hit, release):
        touch = " ".join([str(x) for x in t])
        hit = " ".join([str(x) for x in h])
        release = " ".join([str(x) for x in r])
        file.write(f"{key},{touch},{hit},{release}\n")
    file.close()


if __name__ == "__main__":
    # Parse flags
    force = False
    plot = False
    new_dataset = False
    print_overlaps = False
    NEW_DATASET_ROOT = None  # don't include the trailing "/"
    while sys.argv[1][0] == "-":
        if sys.argv[1] == "-h":
            usage()
            sys.exit(0)
        if sys.argv[1] == "-f":
            force = True
            print("-f: output overwrite enabled")
        elif sys.argv[1] == "-p":
            plot = True
            print("-p: plot previews enabled")
        elif sys.argv[1] == "-n":
            new_dataset = True
            print("-n: will intelligently create a new dataset")
            assert NEW_DATASET_ROOT is not None, "Must specify value of `NEW_DATASET_ROOT`"
        elif sys.argv[1] == "-o":
            print_overlaps = True
            print("-o: will display information about overlaps at the end of execution")
        else:
            print(f"unknown flag: {sys.argv[1]}")
            sys.exit(2)
        del sys.argv[1]

    files = sys.argv[1:]
    for filename in files:
        filename = filename.rstrip(".wav")
        print(f'-> processing {filename} ...')
        wave_file = wave.open(f"{filename}.wav", "rb")
        frames = read_frame_values_from_wav(wave_file)
        times, keys = read_keys_file(filename)
        sampling_rate = wave_file.getframerate()
        wave_file.close()
        if plot:
            plt.plot(frames)
        keypress_list = []
        for time in times:
            keypress_frames = extract_keypress_frames(time, frames, sampling_rate)
            keypress_list.append(keypress_frames)
            if plot:
                plt.axvline(time * sampling_rate, color="red")
        if new_dataset:
            new_dirname = dirname(filename)
            new_dirname = new_dirname.split("/")[-2:]
            new_dirname = "/".join(new_dirname)
            new_dirname = NEW_DATASET_ROOT + "/" + new_dirname
            Path(new_dirname).mkdir(parents=True, exist_ok=True)
            write_to_csv(keys, keypress_list, filename, new_dirname, force=force)
        else:
            write_to_csv(keys, keypress_list, filename, force=force)
        if plot:
            plt.show()
        wave_file.close()
    if print_overlaps:
        print(f"OVERLAPS: {OVERLAPS}")
        print(f"KEY COUNT: {KEY_COUNT}")
        print(f"RATIO OVERLAPS/KEY_COUNT: {OVERLAPS/KEY_COUNT}")
