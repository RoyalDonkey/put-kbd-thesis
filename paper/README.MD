# Thesis paper

The [default template](https://www.overleaf.com/latex/templates/put-dissertation-template/hphrysdzcbzg)
has been slightly altered to use XeLaTeX instead of vanilla LaTeX. This means
there is native UTF-8 unicode support and no need for `inputenc`, `outputenc`
or `babel` packages. There is a slight chance I messed up something in the
process, but I guess we'll find out while writing.

## Dependencies

- `xelatex`
- `graphviz`
- `python3` + `matplotlib`

The `\input{}` directive has been replaced with the `subfiles` package to enable
compiling individual chapters. In doing so you may run into errors due to
`ppfcmthesis.cls` not being found in global path. A dirty, but simple solution
is to create a symlink in the system directory, like so:

    ln -s "$(realpath ppfcmthesis.cls)" /usr/share/texmf-dist/tex/xelatex/

## Compiling

To compile the thesis, run:

    make

A `thesis.pdf` file should appear in the same directory. If it doesn't, or if
the make process returns an error, inspect the log at `cache/thesis.log`.

Compilation cache is stored in `cache/`. To do a clean build, use:

    make clean all
