"""
This module provides an easy interface for accessing training and test data
from our custom directory structure.
"""
import re
import custom_read_csv as crc
from pathlib import Path
from os.path import basename
from typing import Literal

REGEX_TRAIN_CSV = re.compile(r'^train.*\.csv$')
REGEX_TEST_CSV = re.compile(r'^test.*\.csv$')
REGEX_NOTES = re.compile(r'^notes\.txt$')
REGEX_LETTER = re.compile(r'^[a-z]$')
REGEX_DIGIT = re.compile(r'^[0-9]$')
REGEX_SYMBOL = re.compile(r'^(comma|dash|dot|quote|semicolon|slash|space)$')


def build_kps(data_root: str, data_type: Literal['train'] | Literal['test'], debug: bool = False) -> crc.KeyPressSequence:
    """Recurse through a data directory and create a KeyPressSequence object.

    Arguments:
    `data_root`: path to the root data directory (containing 'letters', 'digits' and 'symbols' subdirectories)
    `data_type`: either 'train' or 'test', depending on which data type is desired
    `debug`: (optional) if set to `True`, debug information is printed on stdout

    Example usage:
        train_df = build_kps('../data', 'train').to_pandas()"""
    assert data_type in ('train', 'test')
    kps = crc.KeyPressSequence()
    path = Path(data_root)
    indent = 0

    def iprint(*args, **kwargs):
        if not debug:
            return
        print(indent * '  ', end='')
        print(*args, **kwargs)

    # To be used on a directory containing CSVs
    def gather_csvs(path: Path):
        nonlocal kps, indent
        regex_csv = REGEX_TRAIN_CSV if data_type == 'train' else REGEX_TEST_CSV
        for f in path.iterdir():
            if not f.is_file():
                iprint(f'ignoring non-file: "{f}"')
                continue
            if regex_csv.match(basename(str(f))):
                iprint(f'adding "{f}"')
                kps += crc.KeyPressSequence(str(f))
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
        nonlocal kps, indent
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

    return kps
