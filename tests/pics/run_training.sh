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
rtt_values=(0.04 0.035 0.03 0.025 0.02 0.015 0.01 0.005 0)
bandwidth_values=(214748364800)
models=("minGPT")

cargo build --release ${OPT_FLAG}
for rtt in "${rtt_values[@]}"; do
    for bandwidth in "${bandwidth_values[@]}"; do
        echo "Setting RTT to $rtt and Bandwidth to $bandwidth in config.toml"

        sed -i "s/^rtt = .*/rtt = $rtt/" xpuremoting/config.toml
        sed -i "s/^bandwidth = .*/bandwidth = $bandwidth/" xpuremoting/config.toml
        for model in "${models[@]}"; do

            echo "Stopping old server instance if any..."
            pkill server || true
            echo "Running: RUST_LOG=warn cargo run server"
            RUST_LOG=warn cargo run --release ${OPT_FLAG} server >/dev/null 2>&1 &

            sleep 5

            echo "Running: RUST_LOG=warn run.sh train/${model}/train.py"
            cd tests/apps || {
                echo "Failed to change directory to tests/apps"
                exit 1
            }
            RUST_LOG=warn ./run.sh train/${model}/train.py 1 64 >"../../tests/pics/degradation/train/${model}_($1)_${rtt}_${bandwidth}.log" 2>&1
            cd ../..

            echo "done ---"
        done

    done
done

echo "All operations completed."
