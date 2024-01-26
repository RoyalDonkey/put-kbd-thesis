#!/usr/bin/env python3
"""
This module provides a function to iterate over .csv files in directory structure
and combine them into one .csv file and second one to read this file
"""
import re
from pathlib import Path
from os.path import basename
import pandas as pd

import sys
sys.path.append("../pymodules")
import custom_read_csv as crc

REGEX_CSV = re.compile(r'^.*\.csv$')
REGEX_NOTES = re.compile(r'^notes\.txt$')
REGEX_LETTER = re.compile(r'^[a-z]$')
REGEX_DIGIT = re.compile(r'^[0-9]$')
REGEX_SYMBOL = re.compile(r'^(comma|dash|dot|quote|semicolon|slash|space)$')


def path_name_to_session(path_name):
    file_name = path_name.split("/")[-1]
    name = file_name.split(".")
    name = ".".join(name[:-1])
    return name


def merge_files(data_root: str, output_file: str = "merged_files.csv", debug: bool = False) -> crc.KeyPressSequence:
    """Recurse through a data directory and create a merged .csv file with all data combined.

    Arguments:
    `data_root`: path to the root data directory (containing 'letters', 'digits' and 'symbols' subdirectories)
    `output_file`: where the .csv file should be created
    `debug`: (optional) if set to `True`, debug information is printed on stdout

    Example usage:
        merged_files('../data', 'merged_data.csv')"""
    df = pd.DataFrame(columns=["key", "key_ascii", "touch", "hit", "release", "session"])
    path = Path(data_root)
    indent = 0

    def iprint(*args, **kwargs):
        if not debug:
            return
        print(indent * '  ', end='')
        print(*args, **kwargs)

    # To be used on a directory containing CSVs
    def gather_csvs(path: Path):
        nonlocal df, indent
        for f in path.iterdir():
            if not f.is_file():
                iprint(f'ignoring non-file: "{f}"')
                continue
            if REGEX_CSV.match(basename(str(f))):
                iprint(f'adding "{f}"')
                df2 = crc.KeyPressSequence(f).to_pandas()
                df2.insert(len(df2.columns), "session", path_name_to_session(str(f)))
                df2.insert(1, "key_ascii", None)
                df2["key_ascii"] = df2["key"].apply(ord)
                df = pd.concat([df, df2], ignore_index=True)
                continue
            if REGEX_NOTES.match(basename(str(f))):
                iprint('found note:')
                indent += 1
                with open(str(f), 'r') as note:
                    iprint(note.read().strip('\n'))
                indent -= 1
                continue
            iprint(f'ignoring file: "{f}"')
        indent -= 1

    # To be used on a directory containing subdirectories of CSVs
    def gather_leaf_dirs(path: Path, regex: re.Pattern):
        nonlocal df, indent
        for f in path.iterdir():
            if not f.is_dir():
                iprint(f'ignoring non-directory: "{f}"')
                continue
            if not regex.match(basename(str(f))):
                iprint(f'ignoring non-matching directory: "{f}"')
                continue
            iprint(f'entering {f}')
            indent += 1
            gather_csvs(f)
        indent -= 1

    def write_df_rows_to_csv(df: pd.DataFrame, path: str):
        assert df.columns.to_list() == [
            "key_ascii", "touch", "hit", "release", "session"
        ]
        fp = open(path, "w")
        fp.write(','.join(df.columns.to_list()) + "\n")
        for _, row in df.iterrows():
            csv_row = str(row["key_ascii"]) + ','
            csv_row += ' '.join([str(x) for x in row["touch"]]) + ','
            csv_row += ' '.join([str(x) for x in row["hit"]]) + ','
            csv_row += ' '.join([str(x) for x in row["release"]]) + ','
            csv_row += row["session"]
            fp.write(csv_row + "\n")

    for f in path.iterdir():
        if not f.is_dir():
            iprint(f'ignoring non-directory: "{f}"')
            continue
        if basename(str(f)) == 'letters':
            iprint(f'entering {f}')
            indent += 1
            gather_leaf_dirs(f, REGEX_LETTER)
            continue
        if basename(str(f)) == 'digits':
            iprint(f'entering {f}')
            indent += 1
            gather_leaf_dirs(f, REGEX_DIGIT)
            continue
        if basename(str(f)) == 'symbols':
            iprint(f'entering {f}')
            indent += 1
            gather_leaf_dirs(f, REGEX_SYMBOL)
            continue
        iprint(f'ignoring unrecognized directory: "{f}"')

    df.sort_values(by="key")
    # to avoid problems with characters like space or comma and allow
    # for the format to be used with less common characters
    del df["key"]
    write_df_rows_to_csv(df, output_file)


def read_merged(file_name: str):
    # wontdo: adjust to new format
    # this function is never used and the potential for it being useful never came up
    """
    Read merged to pd.DataFrame

    Arguments:
        `file_name`: which file to read
    """
    def str_to_int_list(a):
        a = a[1:-1].split()
        a = [int(i) for i in a]
        return a

    df = pd.read_csv(file_name, index_col=0)
    df["touch"] = df["touch"].apply(str_to_int_list)
    df["hit"] = df["hit"].apply(str_to_int_list)
    df["release"] = df["release"].apply(str_to_int_list)
    return df


if __name__ == "__main__":
    merge_files("../data")
