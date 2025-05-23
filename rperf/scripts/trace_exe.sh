#!/bin/bash

SCRIPT_DIR=$(dirname $(realpath $0))
ORIGINAL_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
ORIGINAL_REMOTING_BOTTOM_LIBRARY=$REMOTING_BOTTOM_LIBRARY
ORIGINAL_LOG_BREAKPOINT_LIBRARY=$LOG_BREAKPOINT_LIBRARY

CUDA_VISIBLE_DEVICES=0
REMOTING_BOTTOM_LIBRARY=$(realpath $SCRIPT_DIR/../libs/vanilla/output/lib64/libcuda_hook.so)
LOG_BREAKPOINT_LIBRARY=$(realpath $SCRIPT_DIR/../libs/log_breakpoint/log_breakpoint.so)
RUN_INTERCEPTION_SH=$(realpath $SCRIPT_DIR/run_interception.sh)

export CUDA_VISIBLE_DEVICES
export REMOTING_BOTTOM_LIBRARY
export LOG_BREAKPOINT_LIBRARY

if [ -z "$RPERF_OUTPUT_DIR" ]; then
  echo "RPERF_OUTPUT_DIR is not set"
  exit 1
fi

# Run the command
$RUN_INTERCEPTION_SH $@
mv vanilla_rperf.log $RPERF_OUTPUT_DIR

if [ -z "$ORIGINAL_CUDA_VISIBLE_DEVICES" ]; then
  unset CUDA_VISIBLE_DEVICES
else
  export CUDA_VISIBLE_DEVICES=$ORIGINAL_CUDA_VISIBLE_DEVICES
fi

if [ -z "$ORIGINAL_REMOTING_BOTTOM_LIBRARY" ]; then
  unset REMOTING_BOTTOM_LIBRARY
else
  export REMOTING_BOTTOM_LIBRARY=$ORIGINAL_REMOTING_BOTTOM_LIBRARY
fi

if [ -z "$ORIGINAL_LOG_BREAKPOINT_LIBRARY" ]; then
  unset LOG_BREAKPOINT_LIBRARY
else
  export LOG_BREAKPOINT_LIBRARY=$ORIGINAL_LOG_BREAKPOINT_LIBRARY
fi
