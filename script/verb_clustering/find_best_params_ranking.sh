#!/bin/bash

source_dir=./source/verb_clustering
data_dir=./data/verb_clustering

settings=(all_3_0 all_3_1 all_3_2)

pretrained_model_name=bert-base-uncased

model_names=(vanilla arcface_classification softmax_classification)
run_numbers=(00)

# model_names=(siamese_distance adacos_classification triplet_distance)
# model_names=(siamese_distance)
# model_names=(triplet_distance)
# model_names=(arcface_classification)
# run_numbers=(00 01 02 03)

vec_types=(word mask wm)

ranking_methods=(all-all same-same diff-diff)

for setting in "${settings[@]}"; do
    for model_name in "${model_names[@]}"; do
        for vec_type in "${vec_types[@]}"; do
            for ranking_method in "${ranking_methods[@]}"; do
                d1=${setting}
                d2=${pretrained_model_name}/${model_name}
                d3=${vec_type}
                d4=${ranking_method}
                uv run python ${source_dir}/find_best_params_ranking.py \
                    --input_dir ${data_dir}/embedding/"${d1}"/"${d2}" \
                    --output_dir ${data_dir}/best_params_ranking/"${d1}"/"${d2}"/"${d3}"/"${d4}" \
                    --vec_type "${vec_type}" \
                    --run_numbers "${run_numbers[@]}" \
                    --ranking_method "${ranking_method}"
            done
        done
    done
done
