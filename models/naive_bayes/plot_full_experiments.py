#!/usr/bin/env python3
"""Module that creates plots to compare differences of naive bayes classyfier,
if input data is Re part of FFT or its magnitude. Assumes that .json file was
Created using full_experiments.py script"""

import json
import matplotlib.pyplot as plt

RESULTS_FILE = "results/shuffled.json"
FRAMES_IN_PEAK = 176


def create_x_axis(data: dict):
    """Parse json to extract series of x values.U
    Useful when using top/bottom_n or top/bottom_cutoff.
    The function extract x axis from dict with follownig
    structures: data[function_name-mag=Bool]["thr"],
    which is a list of dicts with following
    data ["parameters"]["others"][name_of_values_of_x_series]

    Args:
        data (dict): dict from .json file created from full_experiment.py

    Returns:
        axis_dict (dict): {name_of_function: {name_of_axis: [following_values_of_x]}}
    """
    axis_dict = dict()
    for func_name, func_dict in data.items():
        if "raw" in func_name or "plain_fft" in func_name:
            continue
        if func_name not in axis_dict.keys():
            axis_dict[func_name.split("-")[0]] = dict()
        for el in func_dict["thr"]:
            specyfic_parameters = el["parameters"]["others"]
            assert len(specyfic_parameters.keys()) == 1, "Too many specyfic parameters to create 1D axis"
            key = next(iter(specyfic_parameters.keys()))  # fast way to get first key from dict
            if key not in axis_dict[func_name.split("-")[0]].keys():
                axis_dict[func_name.split("-")[0]][key] = [specyfic_parameters[key]]
            else:
                axis_dict[func_name.split("-")[0]][key].append(specyfic_parameters[key])
    return axis_dict


def fetch_series(mode_res: dict):
    """Helper function to fetch accuracies nested inside a dict

    Args:
        mode_res (dict): dict of particular mode with magnitude (name example: top_n-mag=True)

    Returns:
        out (dict): a dict with each combination of touch/hit/release and its
    corresponding list of accuracies
    """
    out = dict()
    for wave_label, wave_res in mode_res.items():
        t = []
        for el in wave_res:
            t.append(el["accuracy"])
        out[wave_label] = t
    return out


if __name__ == "__main__":
    with open(RESULTS_FILE) as f:
        t = f.read()
        data = json.loads(t)

    # derive data from .json file to plot
    x_axis = create_x_axis(data)
    data_to_plot = dict()
    for func_name, func_dict in data.items():
        if "raw" in func_name or "plain_fft" in func_name:
            continue
        func = func_name.split("-")[0]
        if func not in data_to_plot.keys():
            data_to_plot[func] = {"x": next(iter(x_axis[func].values())), "mag_true": None, "mag_false": None}
        mag_value = func_name.split("=")[-1]
        if mag_value == "True":
            data_to_plot[func]["mag_true"] = fetch_series(func_dict)
        elif mag_value == "False":
            data_to_plot[func]["mag_false"] = fetch_series(func_dict)
        else:
            print("Incorrect value of magnitude, derived from name of function/mode")

    for func_name, val in data_to_plot.items():
        fig, axarr = plt.subplots(1, 2)
        fig.suptitle(f"{func_name} comparison for different windows comb", fontsize=15)
        axarr[0].set_title("fft with real numbers")
        axarr[1].set_title("fft with computed magnitude")
        axarr[0].set_ylim([0, 0.6])
        axarr[1].set_ylim([0, 0.6])
        for (wave, mag_true_values), (_, mag_false_values) in zip(val["mag_true"].items(), val["mag_false"].items()):
            axarr[0].plot(val["x"], mag_false_values, label=wave)
            axarr[0].legend(loc="upper left")
            axarr[1].plot(val["x"], mag_true_values, label=wave)
            axarr[1].legend(loc="upper left")

        plt.show()
