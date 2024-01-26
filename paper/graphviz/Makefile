# Makefile for compiling graphviz graphs into PDFs. Nothing fancy.

SRCS := $(wildcard *.gv)
PDFS := $(SRCS:.gv=.pdf)

.PHONY: all clean

all: $(PDFS)

%.pdf: %.gv
	dot -Tpdf $^ -o $@

clean:
	$(RM) -- $(PDFS)
