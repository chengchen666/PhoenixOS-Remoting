# Rperf tool

This is an automated tool to characterize the overhead of disaggregation for arbitrary applications. Rperf will run application multiple times with different configurations and collect the performance metrics. Then using [theoretical models](disaggregation_model/README.md), it will generate the prediction of the overhead of disaggregation on arbitrary network settings.

## Build

```bash
cd path/to/rperf
make -j
```

Make sure that you have installed the following dependencies:

- `Nsight Systems`

## Usage

### 1. Collect the performance metrics

For collecting, we just need to add the Rperf command wrapper to the application command. For instance, assume we run the specific application with the following command:

```bash
python3 inference.py
```

Then we can run the application with Rperf as follows (in the same directory as the original command):

```bash
RPERF_OUTPUT_DIR=out_dir path/to/rperf/scripts/perf.sh python3 inference.py
```

The `perf.sh` script will run the application multiple times with different configurations, and output the performance metrics to the `RPERF_OUTPUT_DIR`, you can refer to [scripts](scripts/) for more details.

### 2. Run the prediction

`work.py` will input the performance metrics and output the prediction of the disaggregation overhead. We use [calc](work.py#L94) function to calculate the overhead on different network settings.

```bash
cd path/to/rperf
python3 work.py $RPERF_OUTPUT_DIR
```

## Acknowlegement

Thanks to the [cuda_hook](https://github.com/Bruce-Lee-LY/cuda_hook) library which can intercept CUDA API calls locally.

We modify the `cuda_hook` library to collect some of the performance metrics of CUDA API. In particular, we also insert probes to the `LaunchKernel-like` functions (cudnn library, etc.) to collect the kernel execution time with `Nsight Systems` profiling results.
