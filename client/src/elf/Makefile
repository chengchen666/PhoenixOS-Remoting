#!/bin/bash
.PHONY: all clean

CC = gcc
CFLAGS = -lelf -lz -O2
SRC = $(wildcard *.c)
OBJ = $(patsubst %.c, %.o, $(SRC))
LIB = libelfctrl.a

all: $(LIB)

$(LIB): $(OBJ)
	ar rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) $< -fPIC -c -o $@

clean:
	rm -f $(OBJ) $(LIB)
