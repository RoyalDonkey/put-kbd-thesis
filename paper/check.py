#!/usr/bin/env python3
# Scans LaTeX build log for warnings and errors and displays them.

import re
import sys

BLACKLIST = [re.compile(x, re.IGNORECASE) for x in (
    r"error",
    r"warning",
)]

WHITELIST = [re.compile(x) for x in (
    r"^Package: infwarerr 2019/12/03 v1\.5 Providing info/warning/error messages \(HO\)$",
    r"^Package caption Warning: \\label without proper reference on input line \d+\.$",
    r"^LaTeX Warning: Reference `.*?' on page \d+",
    r"^LaTeX Warning: There were undefined references\.$",
)]

if __name__ == "__main__":
    log_file = "cache/thesis.log"
    if len(sys.argv) > 1:
        log_file = sys.argv[1]

    try:
        lines = open(log_file, "r").readlines()
    except FileNotFoundError:
        print(f"log file '{log_file}' not found")
        sys.exit(1)

    filtered_lines: list[tuple[int, str]] = []
    for i in range(len(lines)):
        line = lines[i]
        matches_blacklist = any(map(lambda x: x.search(line), BLACKLIST))
        matches_whitelist = any(map(lambda x: x.search(line), WHITELIST))
        if matches_blacklist and not matches_whitelist:
            filtered_lines.append((i + 1, line.strip()))

    if filtered_lines:
        print("------------[ LaTeX COMPILATION WARNINGS/ERRORS ]------------")
        print(f"{log_file}:")
        for i, line in filtered_lines:
            print("{:>4}|  {}".format(i, line))
        print("-------------------------------------------------------------")
        sys.exit(1)
