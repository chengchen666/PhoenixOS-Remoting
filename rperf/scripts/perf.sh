#!/bin/bash

ORIGINAL_RPERF_OUTPUT_DIR=$RPERF_OUTPUT_DIR
if [ -z "$RPERF_OUTPUT_DIR" ]; then
  TIME=$(date +"%Y%m%d-%H%M%S")
  RPERF_OUTPUT_DIR="rperf-data/${TIME}/"
  export RPERF_OUTPUT_DIR
fi

if [ ! -d "$RPERF_OUTPUT_DIR" ]; then
  mkdir -p "$RPERF_OUTPUT_DIR"
fi

SCRIPT_DIR=$(dirname $(realpath $0))
TRACE_EXE_SH=$(realpath $SCRIPT_DIR/trace_exe.sh)
TRACE_GPU_TIME_SH=$(realpath $SCRIPT_DIR/trace_gpu_time.sh)
TRACE_KERNEL_GROUP_SH=$(realpath $SCRIPT_DIR/trace_kernel_group.sh)

$TRACE_EXE_SH $@
$TRACE_GPU_TIME_SH $@
$TRACE_KERNEL_GROUP_SH $@

echo "========================================"
echo "Rperf output directory: $RPERF_OUTPUT_DIR"
echo "========================================"
if [ -z "$ORIGINAL_RPERF_OUTPUT_DIR" ]; then
  unset RPERF_OUTPUT_DIR
else
  export RPERF_OUTPUT_DIR=$ORIGINAL_RPERF_OUTPUT_DIR
fi
