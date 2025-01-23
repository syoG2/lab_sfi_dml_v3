#!/bin/bash

source_dir=./source/verb_clustering_c4
data_dir=./data/verb_clustering_c4

settings=(all_3_0 all_3_1 all_3_2)
# settings=(all_3_0)

pretrained_model_name=bert-base-uncased
# pretrained_model_name=bert-large-uncased
# pretrained_model_name=roberta-base
# pretrained_model_name=roberta-large

# model_names=(adacos_classification softmax_classification vanilla)
# model_names=(adacos_classification)
# model_names=(softmax_classification)
model_names=(vanilla)
# run_numbers=(00)

# model_names=(arcface_classification siamese_distance triplet_distance)
# model_names=(arcface_classification)
# model_names=(siamese_distance)
# model_names=(triplet_distance)
# run_numbers=(00 01 02 03)

vec_types=(mask wm word)
# vec_types=(mask)
# vec_types=(wm)
# vec_types=(word)

clustering_name=twostep
clustering_method1=xmeans
clustering_method2=average

c4_rate=1

add_method=c4first
# add_method=ratio
# add_method=sequential

for setting in "${settings[@]}"; do
    for model_name in "${model_names[@]}"; do
        for vec_type in "${vec_types[@]}"; do
            d1=${setting}/${pretrained_model_name}/${model_name}
            d2=${vec_type}/${clustering_name}-${clustering_method1}-${clustering_method2}

            splits=(dev test)
            for split in "${splits[@]}"; do
                uv run python ${source_dir}/evaluate_clustering.py \
                    --input_file "${data_dir}/clustering/${add_method}/${c4_rate}/${d1}/${d2}/exemplars_${split}.jsonl" \
                    --input_params_file "${data_dir}/clustering/${add_method}/${c4_rate}/${d1}/${d2}/params.json" \
                    --output_dir "${data_dir}/evaluate_clustering_ours/${add_method}/${c4_rate}/${d1}/${d2}" \
                    --split "${split}"
            done
        done
    done
done