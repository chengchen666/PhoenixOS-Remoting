#!/bin/bash

set -e

default_config_file="default_infer.json"
if [ $# -eq 1 ]; then
  config_file="$1"
else
  config_file="$default_config_file"
fi

output_dir=$(jq -r '.output_dir' "$config_file")
readarray -t models < <(jq -r '.models[]' "$config_file")

bert_model_path=$(jq -r '.bert_model_path' "$config_file")
sd_model_path=$(jq -r '.sd_model_path' "$config_file")
gpt_model_path=$(jq -r '.gpt_model_path' "$config_file")
config_path=$(jq -r '.config_path' "$config_file")

declare -A model_params
model_params["BERT"]="1 64 ${bert_model_path}"
model_params["ResNet18_Cifar10_95.46"]="1 64"
model_params["STABLEDIFFUSION-v1-4"]="1 1 ${sd_model_path}"
model_params["gpt2"]="1 512 ${gpt_model_path}" 

cd ${BASH_SOURCE[0]%/*}
cd ../.. || {
    echo "Failed to change directory to root path"
    exit 1
}


for model in "${models[@]}"; do
    echo "Running ${model}"
    params=${model_params[$model]}
    cd tests/apps || {
        echo "Failed to change directory to tests/apps"
        exit 1
    }
    if [ ! -d "../../${output_dir}" ]; then
        mkdir -p "../../${output_dir}"
    fi
    python3 infer/${model}/inference.py ${params} >"../../${output_dir}/${model}_infer.log" 2>&1
    cd ../..

    echo "done ---"

done