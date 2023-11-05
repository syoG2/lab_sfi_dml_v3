#!/bin/bash

source_dir=../../source/verb_clustering
data_dir=../../data/verb_clustering

# settings=(all_3_0 all_3_1 all_3_2)
settings=(all_3_1)

pretrained_model_name=bert-base-uncased
# model_names=(vanilla softmax_classification adacos_classification siamese_distance triplet_distance arcface_classification)
model_names=(triplet_distance)

# vec_types=(word mask wm)
# vec_types=(word)
# vec_types=(mask)
vec_types=(wm)

clustering_name=onestep
clustering_method=average

for setting in ${settings[@]}; do
    for model_name in ${model_names[@]}; do
        for vec_type in ${vec_types[@]}; do
            d1=${setting}/${pretrained_model_name}/${model_name}
            d2=${vec_type}/${clustering_name}-${clustering_method}
            python ${source_dir}/perform_clustering.py \
                --input_dir ${data_dir}/embedding/${d1} \
                --output_dir ${data_dir}/clustering/${d1}/${d2} \
                --input_params_file ${data_dir}/best_params_clustering/${d1}/${d2}/best_params.json \
                --clustering_name ${clustering_name} \
                --clustering_method ${clustering_method}
        done
    done
done
