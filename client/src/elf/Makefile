#!/bin/bash
.PHONY: all clean

CC = gcc
CFLAGS = -O2
SRC = $(wildcard *.c)
OBJ = $(patsubst %.c, %.o, $(SRC))
LIB = libelf.a

all: $(LIB)

$(LIB): $(OBJ)
	ar rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) $< -c -o $@

clean:
	rm -f $(OBJ) $(LIB)
