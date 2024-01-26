#!/usr/bin/env python3
"""
This module provides `KeyPressData` and `KeyPressSequence` types
as well as some auxiliary functions.

This file can also be run as an executable:

    custom_read_csv.py [FILE...]

This will iterate over csv files, build a `KeyPressSequence` object
from each one and print it to standard output.
"""
import numpy as np
import sys
import pandas as pd
from typing import cast, Optional, Self


def csv_row_to_dict(row: str, header_row: str) -> dict[str, list[int | float | str]]:
    """Returns a dictionary: column name -> list of values.

    Given a row of a csv file and the names of its columns (the header row),
    returns a dictionary associating the column names with row values.
    Each value in the dictionary is a list (they can be empty).
    Numeric values are converted, other remain as strings.
    """
    row_data = row.split(',')
    column_names = header_row.split(',')
    column_names = [column_name.strip() for column_name in column_names]
    data_dict: dict[str, list[int | float | str]] = dict(
        [(column_name, []) for column_name in column_names]
    )
    for column_value, column_name in zip(row_data, column_names):
        space_split = column_value.split()
        for value_s in space_split:
            try:
                value: int | float
                value_i = int(value_s)
                value_f = float(value_s)
                if abs(value_i - value_f) < 0.0000001:
                    value = value_i
                else:
                    value = value_f
                data_dict[column_name].append(value)
            except ValueError:
                data_dict[column_name].append(value_s)
    return data_dict


class KeyPressData():
    """Stores the pressed key, and np.arrays of wav values for touch, hit, and release peaks."""
    def __init__(self, row: str, session: str | None = None):
        row_data = row[2:].split(',')
        self.key = row[0]
        touch = np.array([int(x) for x in row_data[0].split()])
        hit = np.array([int(x) for x in row_data[1].split()])
        release = np.array([int(x) for x in row_data[2].split()])
        assert len(touch) == len(hit) == len(release)
        self.touch = touch
        self.hit = hit
        self.release = release
        if session:
            self.session = session

    def __str__(self):
        """NOTE: this does not include the instance's session
        """
        touch_str = str(self.touch)
        hit_str = str(self.hit)
        release_str = str(self.release)
        return f"key: '{self.key}' ({ord(self.key)})\ntouch: {touch_str}\n" + \
               f"hit: {hit_str}\nrelease: {release_str}"

    def __len__(self):
        return len(self.touch)

    def __eq__(self, __value: object):
        assert isinstance(__value, KeyPressData)
        return all([
            self.key == __value.key,
            np.array_equal(self.touch, __value.touch),
            np.array_equal(self.hit, __value.hit),
            np.array_equal(self.release, __value.release),
        ])


class KeyPressSequence():
    """Wrapper for a list of KeyPressData objects.

    If `file_path` is given, the sequence is loaded from the file.
    Otherwise, an empty object is initialized, allowing for manual instantiation
    of its data members.
    """
    def __init__(self, file_path: Optional[str] = None):
        self.data = []
        if file_path is not None:
            lines: Optional[list[str]] = None
            with open(file_path) as file:
                lines = file.readlines()
            column_names = [name.strip() for name in lines[0].split(',')]
            if column_names == ["key", "touch", "hit", "release"]:
                self.data = [KeyPressData(row) for row in lines[1:]]
            elif column_names == ["key_ascii", "touch", "hit", "release", "session"]:
                self.data = [KeyPressData(*self.session_csv_row_to_kpdtuple(row))
                             for row in lines[1:]]
            else:
                error_msg = "Invalid column names! Expected:\n"
                error_msg += "key, touch, hit, release\nor:\n"
                error_msg += "key_ascii, touch, hit, release, session\n"
                raise ValueError(error_msg)

    def __str__(self):
        str_list = [str(key_data) for key_data in self.data]
        return "\n".join(str_list)

    def __len__(self):
        return len(self.data)

    def __add__(self, kps) -> Self:
        """Concatenates two KeyPressSequence objects."""
        assert isinstance(kps, KeyPressSequence)
        concatenated = KeyPressSequence()
        if self.data and kps.data:
            # check if the added sequences both include .session in their kpds
            try:
                self.data[0].session
                try:
                    kps.data[0].session  # kpds of both kpss include sessions
                except AttributeError:
                    warning = "[WARNING]: The KeyPressSequence being ADDED TO includes"
                    warning += " sessions in its data, while the one BEING ADDED does NOT"
                    print(warning)
            except AttributeError:
                try:
                    kps.data[0].session
                    warning = "[WARNING]: The KeyPressSequence being ADDED TO does NOT"
                    warning += " include sessions in its data, while the one BEING ADDED does"
                    print(warning)
                except AttributeError:
                    pass  # kpds of both kpss DON'T include sessions
        concatenated.data = self.data + kps.data
        return cast(Self, concatenated)  # cast shouldn't be needed, but mypy complains???

    def __getitem__(self, idx) -> KeyPressData:
        return self.data[idx]

    def to_pandas(self) -> pd.DataFrame:
        """Returns a new pandas dataframe containing the same data."""
        return pd.DataFrame(
            [(x.key, x.touch, x.hit, x.release) for x in self.data],
            columns=("key", "touch", "hit", "release")
        )

    @staticmethod
    def session_csv_row_to_kpdtuple(row: str) -> tuple[str, str]:
        """Converts a row of a csv file with columns: "key_ascii", "touch",
        "hit", "release", "session" to a tuple of two elements:
        - a string from which a KeyPressData instance can be constructed ("key", "touch", "hit", "release")
        - the recording session this recording came from
        This is needed to allow KeyPressData to service two file formats
        (more exactly - for it to support the original, and the new,
        session-including one), without compromising memory or too much
        performance.
        """
        key_ascii, touch, hit, release, session = row.split(',')
        key = chr(int(key_ascii))
        session = session.strip()
        return ','.join((key, touch, hit, release)), session


if __name__ == "__main__":
    file_paths = sys.argv[1:]
    for file_path in file_paths:
        kps = KeyPressSequence(file_path)
        print(kps)
