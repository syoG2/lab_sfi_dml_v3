#!/bin/bash

source_dir=./source/verb_clustering
data_dir=./data/verb_clustering

settings=(all_3_0 all_3_1 all_3_2)
# settings=(all_3_0)

pretrained_model_name=bert-base-uncased
# vec_types=(word mask)
# vec_types=(word)
vec_types=(mask)

# model_name=vanilla
# model_name=softmax_classification
# model_name=adacos_classification
# run_numbers=(00)

# model_name=arcface_classification
# model_name=siamese_distance
model_name=triplet_distance
run_numbers=(00 01 02 03)

splits=(train dev test)
# splits=(train)
# splits=(dev)
# splits=(test)

device=cuda:0

for setting in "${settings[@]}"; do
    for vec_type in "${vec_types[@]}"; do
        for run_number in "${run_numbers[@]}"; do
            for split in "${splits[@]}"; do
                d1=${setting}
                d2=${pretrained_model_name}/${model_name}/${vec_type}/${run_number}
                uv run python ${source_dir}/get_embedding.py \
                    --input_file "${data_dir}/dataset/${d1}/exemplars_${split}.jsonl" \
                    --input_params_file "${data_dir}/train_model/${d1}/${d2}/params.json" \
                    --input_model_file "${data_dir}/train_model/${d1}/${d2}/pretrained_model_last.pth" \
                    --output_dir "${data_dir}/embedding/${d1}/${d2}"
            done
        done
    done
done
