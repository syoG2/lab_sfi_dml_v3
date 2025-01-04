#!/bin/bash

source_dir=./source/verb_clustering
data_dir=./data/verb_clustering

# settings=(all_3_0 all_3_1 all_3_2)
settings=(all_3_2)

pretrained_model_name=bert-base-uncased
# model_names=(vanilla softmax_classification adacos_classification siamese_distance triplet_distance arcface_classification)
model_names=(vanilla softmax_classification adacos_classification)
model_names=(triplet_distance)

vec_types=(word mask wm)
# vec_types=(word)
# vec_types=(mask)
# vec_types=(wm)

# clustering_name_method=(onestep-average twostep-xmeans-average)
# clustering_name_method=(onestep-average)
clustering_name_method=(twostep-xmeans-average)

splits=(dev test)

for setting in "${settings[@]}"; do
    for model_name in "${model_names[@]}"; do
        for vec_type in "${vec_types[@]}"; do
            for split in "${splits[@]}"; do
                d1=${setting}/${pretrained_model_name}/${model_name}
                d2="${vec_type}/${clustering_name_method}"
                uv run python ${source_dir}/evaluate_clustering.py \
                    --input_file "${data_dir}/clustering/${d1}/${d2}/exemplars_${split}.jsonl" \
                    --input_params_file "${data_dir}/clustering/${d1}/${d2}/params.json" \
                    --output_dir "${data_dir}/evaluate_clustering_ours/${d1}/${d2}" \
                    --split "${split}"
            done
        done
    done
done
