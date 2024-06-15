#!/bin/bash
LD_LIBRARY_PATH=../../target/release:$LD_LIBRARY_PATH \
LD_PRELOAD=../../target/release/libclient.so:$LD_PRELOAD \
python3 $1 $2 $3 $4