CC=cc
LINKER=cc
CFLAGS=-O2 -std=c99 -Wall -Wextra -pedantic
LDFLAGS=-lSDL2

# All SRCDIR subdirectories that contain source files
DIRS=. tinywav

SRCDIR=src
OBJDIR=obj
SRCDIRS:=$(foreach dir, $(DIRS), $(addprefix $(SRCDIR)/, $(dir)))
OBJDIRS:=$(foreach dir, $(DIRS), $(addprefix $(OBJDIR)/, $(dir)))
SRCS:=$(foreach dir, $(SRCDIRS), $(wildcard $(dir)/*.c))
OBJS:=$(patsubst $(SRCDIR)/%, $(OBJDIR)/%, $(SRCS:.c=.o))
TARGET=recdata

.PHONY: directories all main clean debug

all: directories main

directories:
	mkdir -p $(SRCDIRS) $(OBJDIRS)

main: $(OBJS)
	$(LINKER) $(OBJS) $(LDFLAGS) -o $(TARGET)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) -c $(CFLAGS) $^ -o $@

clean:
	$(RM) -- $(OBJS)

debug: CFLAGS += -g -Og
debug: clean all
