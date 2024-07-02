#!/bin/bash
setups=("cxl-1000") # you can choose the setups, which can be defined manually in rtts and bandwidths part
output_dir="output-dir" # output path under tests/performace directory
# "BERT-pytorch" "ResNet18_Cifar10_95.46" "naifu-diffusion"
models=("BERT-pytorch" "ResNet18_Cifar10_95.46" "naifu-diffusion")
# config file, using default path
config_path="xpuremoting/config.toml"


set -e
cd ${BASH_SOURCE[0]%/*}
cd ../.. || {
    echo "Failed to change directory to root path"
    exit 1
}

if [ $# -eq 0 ]; then
    echo "usage: $0 [opt|raw]"
    exit 1
fi

if [ "$1" == "opt" ]; then
    OPT_FLAG="--features async_api,shadow_desc,local"
elif [ "$1" == "raw" ]; then
    OPT_FLAG=""
else
    echo "invalid param: $1"
    echo "usage: $0 [opt|raw]"
    exit 1
fi

declare -A model_params
model_params["BERT-pytorch"]="1 64"
model_params["ResNet18_Cifar10_95.46"]="1 64"
model_params["naifu-diffusion"]="1 1"

# rtt in ms
declare -A rtts
rtts["cxl-1000"]=0.0002
rtts["cxl-200"]=0.0002
rtts["cxl-40"]=0.0002
rtts["rdma-inrack-200"]=0.005
rtts["rdma-inrack-40"]=0.005
rtts["rdma-inrack-10"]=0.005
rtts["rdma-middle-200"]=0.01
rtts["rdma-middle-40"]=0.01
rtts["rdma-middle-10"]=0.01
# bandwidth in bps
declare -A bandwidths
bandwidths["cxl-1000"]=1073741824000
bandwidths["cxl-200"]=214748364800
bandwidths["cxl-40"]=42949672960
bandwidths["rdma-inrack-200"]=214748364800
bandwidths["rdma-inrack-40"]=42949672960
bandwidths["rdma-inrack-10"]=10737418240
bandwidths["rdma-middle-200"]=214748364800
bandwidths["rdma-middle-40"]=42949672960
bandwidths["rdma-middle-10"]=10737418240
cargo build --release ${OPT_FLAG}

for model in "${models[@]}"; do
    for setup in "${setups[@]}"; do
        rtt=${rtts[$setup]}
        bandwidth=${bandwidths[$setup]}
        # you can redefine output_dir according to setup and other configs
        # output_dir=${output_dir}/${setup}
        params=${model_params[$model]}
        echo "Setting RTT to $rtt and Bandwidth to $bandwidth in config.toml"

        sed -i "s/^rtt = .*/rtt = $rtt/" ${config_path}
        sed -i "s/^bandwidth = .*/bandwidth = $bandwidth/" ${config_path}

        echo "Stopping old server instance if any..."
        pkill server || true

        echo "Start server"
        RUST_LOG=warn cargo run --release ${OPT_FLAG} server >/dev/null 2>&1 &
        sleep 2

        echo "Running: RUST_LOG=warn run.sh train/${model}/train.py"
        cd tests/apps || {
            echo "Failed to change directory to tests/apps"
            exit 1
        }
        if [ ! -d "../../tests/performance/${output_dir}" ]; then
            mkdir -p "../../tests/performance/${output_dir}"
        fi
        RUST_LOG=warn ./run.sh train/${model}/train.py ${params} >"../../tests/performance/${output_dir}/${model}_train_($1)_${rtt}_${bandwidth}.log" 2>&1
        cd ../..

        echo "done ---"
    done
done
pkill server
echo "All operations completed."
