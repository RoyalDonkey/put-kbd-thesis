#!/usr/bin/env python3

# Generates tables for Appendix B.
# Prints the result on stdout; redirect to save to a file.

import os
import json
from collections import OrderedDict

JSON_DIR = "../plot/results"

TABLE_HEAD = \
"""\\begin{table}[h]
\t\\centering
\t\\caption{CAPTION}
\t\\vspace{-0.15cm}
\t\\label{tab:LABEL}
\t\\begin{small}
\t\\begin{tabular}{|r|c|c|c|c|c|c|}
\t\t\\hline"""

TABLE_HEAD_VSPACE = \
"""\\begin{table}[h]
\t\\centering
\t\\vspace{-0.12cm}
\t\\caption{CAPTION}
\t\\vspace{-0.15cm}
\t\\label{tab:LABEL}
\t\\begin{small}
\t\\begin{tabular}{|r|c|c|c|c|c|c|}
\t\t\\hline"""

TABLE_TAIL = \
"""\t\\end{tabular}
\t\\end{small}
\\end{table}"""

TABLE_MODELS_HEAD = \
"""\t\t & \\multicolumn{3}{|c|}{\\textbf{MODEL1}} & \\multicolumn{3}{|c|}{\\textbf{MODEL2}} \\\\"""

TABLE_MODELS_PEAKS_HEAD = \
"""\t\t & \\textbf{none} & \\textbf{FFT} & \\textbf{MFCC} & \\textbf{none} & \\textbf{FFT} & \\textbf{MFCC} \\\\"""

TABLE_MODELS_ROW = \
"""\t\t{peak} & {m1_none} & {m1_fft} & {m1_mfcc} & {m2_none} & {m2_fft} & {m2_mfcc} \\\\"""

TABLE_SEP = "\t\t\\hline"

# Implies 3 rows, because in total there are 6 types of models.
# This is not a parameter; it exists only to avoid using magic numbers.
MODEL_COLS = 2

PEAKS_ORDER = ["thr", "th-", "t-r", "-hr", "t--", "-h-", "--r"]
PRECISION = 7
MODEL_NAME_MAP = {
    "knn_1": "1-NN",
    "logistic_regression": "Logistic Regression",
    "naive_bayes": "Naive Bayes",
    "rnn": "RNN",
    "svm": "SVM",
    "xgboost": "XGBoost",
}

ClassifierResults = dict[str, dict[str, dict[str, dict[str, float]]]]

def print_table(data_list: list[tuple[str, ClassifierResults]], train: str, test: str, vspace: bool = False) -> None:
    train_sanitized = train.replace("_", "\\_")
    test_sanitized = test.replace("_", "\\_")
    print((TABLE_HEAD if not vspace else TABLE_HEAD_VSPACE)
          .replace("CAPTION", f"Model accuracies (train: {train_sanitized}, test: {test_sanitized})")
          .replace("LABEL", f"model_acc_{train}_{test}"))
    print(TABLE_MODELS_PEAKS_HEAD)
    print(TABLE_SEP)
    for i in range(len(data_list) // MODEL_COLS):
        model1_data = data_list[2 * i]
        model2_data = data_list[2 * i + 1]

        model1_name = MODEL_NAME_MAP[model1_data[0]]
        model2_name = MODEL_NAME_MAP[model2_data[0]]

        print(TABLE_MODELS_HEAD
              .replace("MODEL1", model1_name)
              .replace("MODEL2", model2_name))
        print(TABLE_SEP)

        # Only consider results obtained from testing on the same dataset as training
        testset1_data = model1_data[1][train][test]
        testset2_data = model2_data[1][train][test]
        for peak_type in PEAKS_ORDER:
            print(TABLE_MODELS_ROW.format(
                peak=f"\\texttt{{{peak_type}}}",
                m1_none=round(testset1_data["raw"][peak_type], PRECISION),
                m1_fft=round(testset1_data["fft"][peak_type], PRECISION),
                m1_mfcc=round(testset1_data["mfcc"][peak_type], PRECISION),
                m2_none=round(testset2_data["raw"][peak_type], PRECISION),
                m2_fft=round(testset2_data["fft"][peak_type], PRECISION),
                m2_mfcc=round(testset2_data["mfcc"][peak_type], PRECISION),
            ))
        print(TABLE_SEP)
    print(TABLE_TAIL)

if __name__ == "__main__":
    data = OrderedDict[str, ClassifierResults]()
    for fname in os.listdir(JSON_DIR):
        fpath = f"{JSON_DIR}/{fname}"
        fname_extless = os.path.splitext(fname)[0]
        data[fname_extless] = json.load(open(fpath, "r"))
    assert len(data) == 6
    data_list = list[tuple[str, ClassifierResults]](data.items())
    del data


    datasets = ["main", "g213", "k3p", "matebook14", "CA_tb14", "all"]
    i = 0
    for train_dataset in datasets:
        for test_dataset in datasets:
            # Comment this out to generate ALL tables
            if train_dataset != test_dataset:
                continue

            print_table(data_list, train_dataset, test_dataset, vspace=(i % 2 == 0))
            print()
            i += 1
