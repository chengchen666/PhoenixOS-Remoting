#!/bin/bash

set -e

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

cd /workspace || {
  echo "Failed to change directory to /workspace"
  exit 1
}
# 0.04 0.035 0.03 0.025 0.02 0.015 0.01 0.005 0
rtt_values=(0.021)
bandwidth_values=(214748364800)
batch_size=64
# rtt_values=(0.0034)
# bandwidth_values=(214748364800)
# "BERT" "gpt2" "ResNet18_Cifar10_95.46" "STABLEDIFFUSION-v1-4"
models=("gpt2")

declare -A model_params
model_params["BERT"]="1 ${batch_size} /workspace/tests/apps/infer/BERT/bert-base-uncased"
model_params["ResNet18_Cifar10_95.46"]="1 ${batch_size}"
model_params["STABLEDIFFUSION-v1-4"]="1 ${batch_size} /workspace/tests/apps/infer/STABLEDIFFUSION-v1-4/stable-diffusion-v1-4"
model_params["gpt2"]="1 ${batch_size} /workspace/tests/apps/infer/gpt2/gpt2"
cargo build --release ${OPT_FLAG}

for rtt in "${rtt_values[@]}"; do
  for bandwidth in "${bandwidth_values[@]}"; do
    echo "Setting RTT to $rtt and Bandwidth to $bandwidth in config.toml"

    sed -i "s/^rtt = .*/rtt = $rtt/" xpuremoting/config.toml
    sed -i "s/^bandwidth = .*/bandwidth = $bandwidth/" xpuremoting/config.toml
    for model in "${models[@]}"; do
      params=${model_params[$model]}

      echo "Stopping old server instance if any..."
      pkill server || true

      echo "Running: RUST_LOG=warn cargo run server"
      RUST_LOG=warn cargo run --release ${OPT_FLAG} server >/dev/null 2>&1 &

      sleep 5

      echo "Running: RUST_LOG=warn run.sh infer/${model}/inference.py"
      cd tests/apps || {
        echo "Failed to change directory to tests/apps"
        exit 1
      }
      RUST_LOG=warn ./run.sh infer/${model}/inference.py ${params} >"../../tests/performance/degradation/infer/${model}_($1)_${batch_size}_${rtt}_${bandwidth}.log" 2>&1
      cd ../..

      echo "done ---"

    done

  done
done

echo "All operations completed."
