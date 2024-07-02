#!/bin/bash
# Support inference tracking now, you can modify it to training version based on 'run_training' and 'run_training_cfg'
set -e

rtt_values=(0.001 0.002) # list of rtts
bandwidth_values=(77491947438.08) # list of bandwidths
output_dir="output-dir" 
# "BERT" "ResNet18_Cifar10_95.46" "gpt2" "STABLEDIFFUSION-v1-4"
models=("BERT" "ResNet18_Cifar10_95.46" "gpt2" "STABLEDIFFUSION-v1-4")
# if you download the model, you can set model path here, otherwise the program will use online model
bert_model_path=""
sd_model_path=""
gpt_model_path=""
# config file, using default path
config_path="xpuremoting/config.toml"
# log path
log_dir="/workspace/log"

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
model_params["BERT"]="1 64 ${bert_model_path}"
model_params["ResNet18_Cifar10_95.46"]="1 64"
model_params["STABLEDIFFUSION-v1-4"]="1 1 ${sd_model_path}" 
model_params["gpt2"]="1 512 ${gpt_model_path}" 
cargo build --release ${OPT_FLAG}

echo "Setting RTT to $rtt and Bandwidth to $bandwidth in config.toml"

for model in "${models[@]}"; do
    params=${model_params[$model]}
    server_file="server_${model}_${rtt}_${bandwidth}.log"
    client_file="client_${model}_${rtt}_${bandwidth}.log"

    echo "Stopping old server instance if any..."
    pkill server || true

    echo "Running server"
    cargo run --release ${OPT_FLAG} server >"${log_dir}/${server_file}" 2>&1 &

    sleep 2

    echo "Running: run.sh infer/${model}/inference.py ${params}"
    cd tests/apps || {
        echo "Failed to change directory to tests/apps"
        exit 1
    }
    ./run.sh infer/${model}/inference.py ${params} >"${log_dir}/${client_file}" 2>&1
    cd ../..

    echo "extract"

    python3 ${log_dir}/extract.py "${log_dir}/${server_file}" "${log_dir}/out_${server_file}"
    python3 ${log_dir}/extract.py "${log_dir}/${client_file}" "${log_dir}/out_${client_file}"

    echo "merge"

    python3 ${log_dir}/merge.py "${log_dir}/out_${client_file}" "${log_dir}/out_${server_file}" "${log_dir}/out_${model}_${batch_size}.log"
    
    echo "done---"

done

echo "All operations completed."
