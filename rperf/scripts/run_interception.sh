#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath $0))

# Run the command
LD_PRELOAD=$(realpath $SCRIPT_DIR/../libs/vanilla/output/lib64/libcuda_hook.so) $@
