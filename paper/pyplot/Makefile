# Makefile for compiling matplotlib plots into PDFs. Nothing fancy.

SRCS := $(wildcard *.py)
.PHONY: all clean

all: $(SRCS)

%.py:
	env python3 $^

clean:
	$(RM) -- *.pdf
