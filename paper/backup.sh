#!/bin/sh

# Script to copy Overleaf's zip contents to the repo.
# Makes it faster to carry over changes and have a backup.
# TODO: Rewrite in python for cross-platform use.

zipfile="${1:-put-kbd-thesis.zip}"

[ ! -f "$zipfile" ] && echo "zip file not found" >&2 && exit 1

tmpd=$(mktemp -d backup.XXXXX)

unzip "$zipfile" -d "$tmpd"
cp -r -- "$tmpd"/* .

rm -rf -- "$tmpd"
