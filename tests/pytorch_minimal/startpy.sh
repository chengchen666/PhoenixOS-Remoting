#!/bin/bash

LD_PRELOAD=../../target/debug/libclient.so:$LD_PRELOAD \
CUDA_VISIBLE_DEVICES=1 \
python3 $1
