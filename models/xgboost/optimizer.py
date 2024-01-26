#!/usr/bin/env python3
# This is a meta-script for optimizing hyperparameters of the XGBoost model.
# It iterates through supplied ranges for multiple hyperparams, trains the
# model, exports models and notes their performance down in README.MD.

import os
import itertools
import json
import xgboost_classifier as xgbc
from typing import Any, Iterable, Literal
from functools import reduce

XGBOOST_PARAMS_SORT_ORDER = [
    "booster",
    "learning_rate",
    "min_split_loss",
    "max_depth",
    "min_child_weight",
    "max_delta_step",
    "subsample",
    "sampling_method",
    "reg_lambda",
    "reg_alpha",
    "tree_method",
    "process_type",
    "grow_policy",
    "max_leaves",
    "predictor",
    "num_parallel_tree",
    "objective",
    "num_class",
    "eval_metric",
    "seed",
]


def frange(begin: int | float, end: int | float, step: int | float) -> list[int | float]:
    """More convenient `range` that accepts floats."""
    ret = []
    i = 1
    val = begin
    while val < end:
        ret.append(val)
        val = begin + i * step
        i += 1
    return ret


# Bear in mind, each evaluation takes at least 45 seconds
param_specs = [
    {
        "learning_rate": frange(0.1, 1.1, 0.1),
        "min_split_loss": frange(0, 0.24, 0.04),
        "max_depth": frange(3, 18, 3),
    },
    {
        "min_child_weight": frange(0.1, 1.0, 0.1),
        "min_split_loss": [0.12, 0.16, 0.2],
    },
]
RUNS: list[dict[Literal["config"] | Literal["param_specs"], Any]]
RUNS = [
    {
        "config": {
            "output_dir": "results/raw_t--",
            "load_xgb_data_kwargs": {"subset": "t"},
        },
        "param_specs": param_specs,
    },
    {
        "config": {
            "output_dir": "results/raw_-h-",
            "load_xgb_data_kwargs": {"subset": "h"},
        },
        "param_specs": param_specs,
    },
    {
        "config": {
            "output_dir": "results/raw_--r",
            "load_xgb_data_kwargs": {"subset": "r"},
        },
        "param_specs": param_specs,
    },
    {
        "config": {
            "output_dir": "results/raw_th-",
            "load_xgb_data_kwargs": {"subset": "th"},
        },
        "param_specs": param_specs,
    },
    {
        "config": {
            "output_dir": "results/raw_t-r",
            "load_xgb_data_kwargs": {"subset": "tr"},
        },
        "param_specs": param_specs,
    },
    {
        "config": {
            "output_dir": "results/raw_-hr",
            "load_xgb_data_kwargs": {"subset": "hr"},
        },
        "param_specs": param_specs,
    },
    {
        "config": {
            "output_dir": "results/raw_thr",
            "load_xgb_data_kwargs": {"subset": "thr"},
        },
        "param_specs": param_specs,
    },
    {
        "config": {
            "output_dir": "results/fft_t--",
            "load_xgb_data_kwargs": {"preprocessing": "FFT", "subset": "t"},
        },
        "param_specs": param_specs,
    },
    {
        "config": {
            "output_dir": "results/fft_-h-",
            "load_xgb_data_kwargs": {"preprocessing": "FFT", "subset": "h"},
        },
        "param_specs": param_specs,
    },
    {
        "config": {
            "output_dir": "results/fft_--r",
            "load_xgb_data_kwargs": {"preprocessing": "FFT", "subset": "r"},
        },
        "param_specs": param_specs,
    },
    {
        "config": {
            "output_dir": "results/fft_th-",
            "load_xgb_data_kwargs": {"preprocessing": "FFT", "subset": "th"},
        },
        "param_specs": param_specs,
    },
    {
        "config": {
            "output_dir": "results/fft_t-r",
            "load_xgb_data_kwargs": {"preprocessing": "FFT", "subset": "tr"},
        },
        "param_specs": param_specs,
    },
    {
        "config": {
            "output_dir": "results/fft_-hr",
            "load_xgb_data_kwargs": {"preprocessing": "FFT", "subset": "hr"},
        },
        "param_specs": param_specs,
    },
    {
        "config": {
            "output_dir": "results/fft_thr",
            "load_xgb_data_kwargs": {"preprocessing": "FFT", "subset": "thr"},
        },
        "param_specs": param_specs,
    },
    {
        "config": {
            "output_dir": "results/mfcc_t--",
            "load_xgb_data_kwargs": {"preprocessing": "MFCC", "subset": "t"},
        },
        "param_specs": param_specs,
    },
    {
        "config": {
            "output_dir": "results/mfcc_-h-",
            "load_xgb_data_kwargs": {"preprocessing": "MFCC", "subset": "h"},
        },
        "param_specs": param_specs,
    },
    {
        "config": {
            "output_dir": "results/mfcc_--r",
            "load_xgb_data_kwargs": {"preprocessing": "MFCC", "subset": "r"},
        },
        "param_specs": param_specs,
    },
    {
        "config": {
            "output_dir": "results/mfcc_th-",
            "load_xgb_data_kwargs": {"preprocessing": "MFCC", "subset": "th"},
        },
        "param_specs": param_specs,
    },
    {
        "config": {
            "output_dir": "results/mfcc_t-r",
            "load_xgb_data_kwargs": {"preprocessing": "MFCC", "subset": "tr"},
        },
        "param_specs": param_specs,
    },
    {
        "config": {
            "output_dir": "results/mfcc_-hr",
            "load_xgb_data_kwargs": {"preprocessing": "MFCC", "subset": "hr"},
        },
        "param_specs": param_specs,
    },
    {
        "config": {
            "output_dir": "results/mfcc_thr",
            "load_xgb_data_kwargs": {"preprocessing": "MFCC", "subset": "thr"},
        },
        "param_specs": param_specs,
    },
]


def iter_params(param_spec: dict[str, list[Any]]) -> Iterable[dict[str, Any]]:
    for values in itertools.product(*param_spec.values()):
        yield dict(zip(param_spec.keys(), values))


def get_filename(params: dict[str, Any]) -> str:
    words = []
    for k, v in params.items():
        words.append("{}={}".format(
            str(k).replace("_", ""),
            str(v).replace("_", "")
        ))
    return "_".join(words)


def evaluate(fname: str, dtrain: Any, deval: Any, params: dict[str, Any], export_dir: str) -> float:
    evallist = [(deval, "deval")]
    evals_result: dict = {}

    bst = xgbc.train_model(dtrain, evallist, evals_result, params)
    xgbc.export_model(
        bst,
        evals_result,
        os.path.join(export_dir, f"{fname}.model"),
        os.path.join(export_dir, f"{fname}.json")
    )

    acc = xgbc.evaluate(bst, evals_result, deval)
    print(f"----> accuracy: {acc}")
    return acc


def write_readme(fpath: str, config: dict, evaluations: dict) -> None:
    with open(fpath, "w") as f:
        f.write("\n".join([
            "# XGBoost {}".format(", ".join([f"`{k}={repr(v)}`" for k, v in config["load_xgb_data_kwargs"].items()])),
            "",
            "This file was autogenerated from `{}` by `optimizer.py` -- do not change it manually.  ".format(evaluations_fpath),
            "The first row contains \"baseline\" parameters. To improve readability and",
            "reduce redundancy, subsequent rows only include deviations from that baseline:",
            "",
            "<table>",
            "    <tr>",
            "        <th>Accuracy</th>",
            "        <th>Parameters</th>",
            "    </tr>",
        ]))
        f.write("\n")

        def nice_dict(d: dict) -> str:
            legal_params = set(XGBOOST_PARAMS_SORT_ORDER)
            d_params = set(d.keys())
            assert legal_params >= d_params, f"bad param (missing XGBOOST_PARAMS_SORT_ORDER entry?): {d_params - legal_params}"

            return "\n".join([
                "{:<20}: {},".format(f'"{k}"', repr(d[k]).replace("'", '"'))
                for k in XGBOOST_PARAMS_SORT_ORDER if k in d
            ])

        evaluations["runs"].sort(key=lambda x: x["accuracy"], reverse=True)
        for e in [evaluations["baseline"]] + evaluations["runs"]:
            f.write("\n".join([
                "    <tr>",
                "        <td style=\"text-align: center\">{}</td>".format(e["accuracy"]),
                "        <td>",
                "            <pre>",
                "                <code>",
                nice_dict(e["parameters"]),
                "                </code>",
                "            </pre>",
                "        </td>",
                "    </tr>",
            ]))
            f.write("\n")
        f.write("</table>\n")


if __name__ == "__main__":
    base_params = xgbc.configure()

    for i, run in enumerate(RUNS, 1):
        print(f"-> RUN {i}/{len(RUNS)}")

        output_dir = run["config"]["output_dir"]
        models_dir = os.path.join(output_dir, "models")
        readme_fpath = os.path.join(output_dir, "README.MD")
        evaluations_fpath = os.path.join(output_dir, "evaluations.json")
        print(f"--> models:      {models_dir}")
        print(f"--> readme:      {readme_fpath}")
        print(f"--> evaluations: {evaluations_fpath}")

        # Ensure models_dir exists
        os.makedirs(models_dir, exist_ok=True)

        # Read cached evaluations
        try:
            with open(evaluations_fpath) as f:
                evaluations = json.load(f)
        except FileNotFoundError:
            print("--> Evaluations file not found, creating a new one")
            evaluations = {
                "baseline": {},
                "runs": [],
            }

        print("--> Loading XGB data  ({})".format(
            ", ".join([
                f"{k}={repr(v)}"
                for k, v in run["config"]["load_xgb_data_kwargs"].items()
            ])
        ))
        dtrain, deval = xgbc.load_xgb_data(**run["config"]["load_xgb_data_kwargs"])

        print("--> BASELINE")
        if evaluations["baseline"]:
            print("---> Skipping baseline")
        else:
            print("---> Evaluating baseline")
            evaluations["baseline"] = {
                "accuracy": evaluate("baseline", dtrain, deval, base_params, models_dir),
                "parameters": base_params,
            }

        for j, param_spec in enumerate(run["param_specs"], 1):
            print(f"--> RUN {i} PARAM SPEC {j}/{len(run['param_specs'])}")

            # Fancy way to get the product of lengths of param_spec lists
            n_params = reduce(lambda x, y: x * y, map(lambda x: len(x), param_spec.values()))

            k = 1
            for params in iter_params(param_spec):
                # Strip parameters with values equal to base_params'
                to_remove = set()
                for param, value in params.items():
                    if base_params[param] == value:
                        to_remove.add(param)
                for param in to_remove:
                    params.pop(param)

                fname = get_filename(params)

                if any([e["parameters"] == params for e in evaluations["runs"]]):
                    # The result for these params is already in evaluations.json
                    print(f"---> Skipping {k}/{n_params}: {fname.replace('_', ' ')}")
                else:
                    print(f"---> Evaluating {k}/{n_params}: {fname.replace('_', ' ')}")
                    acc = evaluate(fname, dtrain, deval, base_params | params, models_dir)

                    evaluations["runs"].append({
                        "accuracy": acc,
                        "parameters": params,
                    })

                    # Update evaluations.json
                    print(f"----> Updating {evaluations_fpath}")
                    with open(evaluations_fpath, "w") as f:
                        json.dump(evaluations, f, indent=4)

                    # Update README.MD
                    print(f"----> Updating {readme_fpath}")
                    write_readme(readme_fpath, run["config"], evaluations)
                k += 1

            # Update evaluations.json
            print(f"---> Updating {evaluations_fpath}")
            with open(evaluations_fpath, "w") as f:
                json.dump(evaluations, f, indent=4)

            # Update README.MD
            print(f"---> Updating {readme_fpath}")
            write_readme(readme_fpath, run["config"], evaluations)
        else:
            if len(run["param_specs"]) == 0:
                print(f"--> RUN {i} has no param specs -- there is nothing to do")
