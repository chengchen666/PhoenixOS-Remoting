#!/bin/bash
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
config_path="config.toml"


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
model_params["BERT"]="1 64 ${bert_model_path}"
model_params["ResNet18_Cifar10_95.46"]="1 64"
model_params["STABLEDIFFUSION-v1-4"]="1 1 ${sd_model_path}"
model_params["gpt2"]="1 512 ${gpt_model_path}" 
cargo build --release ${OPT_FLAG}

for rtt in "${rtt_values[@]}"; do
  for bandwidth in "${bandwidth_values[@]}"; do
    echo "Setting RTT to $rtt and Bandwidth to $bandwidth in config.toml"

    sed -i "s/^rtt = .*/rtt = $rtt/" ${config_path}
    sed -i "s/^bandwidth = .*/bandwidth = $bandwidth/" ${config_path}
    for model in "${models[@]}"; do
      params=${model_params[$model]}

      echo "Stopping old server instance if any..."
      pkill server || true

      echo "Running: RUST_LOG=warn cargo run server"
      RUST_LOG=warn cargo run --release ${OPT_FLAG} server >/dev/null 2>&1 &

      sleep 2

      echo "Running: RUST_LOG=warn run.sh infer/${model}/inference.py"
      cd tests/apps || {
        echo "Failed to change directory to tests/apps"
        exit 1
      }
      NETWORK_CONFIG=../../config.toml RUST_LOG=warn ./run.sh infer/${model}/inference.py ${params} >"../../tests/performance/${output_dir}/${model}_infer_($1)_${batch_size}_${rtt}_${bandwidth}.log" 2>&1
      cd ../..

      echo "done ---"

    done

  done
done
pkill server
echo "All operations completed."
