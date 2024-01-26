# Compiles the thesis
# In case of errors, check the log file: cache/presentation.log

CACHEDIR = cache
GVDIR = graphviz
PYPLOTDIR = pyplot
LATEX = xelatex
JOBNAME = thesis
LATEXLOG = $(CACHEDIR)/$(JOBNAME).log
LXFLAGS = -interaction batchmode -8bit -shell-escape -output-directory "$(CACHEDIR)" -jobname $(JOBNAME)

.PHONY: directories all graphviz pyplot main clean

all: directories graphviz pyplot main

directories:
	mkdir -p -- $(CACHEDIR)

graphviz:
	$(MAKE) -C $(GVDIR)

pyplot:
	$(MAKE) -C $(PYPLOTDIR)

main: main.tex
	@echo 'Starting build...'
	@echo '-------------------------------------------------------------'
	$(RM) $(LATEXLOG)
	$(LATEX) $(LXFLAGS) -- $^ && \
	biber --output-dir $(CACHEDIR) -- thesis && \
	$(LATEX) $(LXFLAGS) -- $^ && \
	$(LATEX) $(LXFLAGS) -- $^ || true
	@echo '-------------------------------------------------------------'
	
	@echo 'Checking log for errors...'
	@./check.py $(LATEXLOG)
	
	mv -- $(CACHEDIR)/$(JOBNAME).pdf .
	@echo 'Build successful.'

clean:
	$(MAKE) -C $(GVDIR)
	$(RM) -r -- $(JOBNAME).pdf $(CACHEDIR)/
