#!/bin/bash

source_dir=../../source/verb_clustering
data_dir=../../data/verb_clustering

# settings=(all_3_0 all_3_1 all_3_2)
settings=(all_3_2)

pretrained_model_name=bert-base-uncased
model_names=(siamese_distance triplet_distance arcface_classification)
# vec_types=(word mask wm)
# vec_types=(word mask)
vec_types=(wm)

run_numbers=(00 01 02 03)

clustering_name=onestep
clustering_method=average

for setting in ${settings[@]}; do
    for model_name in ${model_names[@]}; do
        for vec_type in ${vec_types[@]}; do
            d1=${setting}/${pretrained_model_name}/${model_name}
            d2=${vec_type}/${clustering_name}/${clustering_method}
            python ${source_dir}/find_best_params_clustering.py \
                --input_dir ${data_dir}/embedding/${d1} \
                --output_dir ${data_dir}/best_params_clustering/${d1}/${d2} \
                --vec_type ${vec_type} \
                --run_numbers ${run_numbers[@]} \
                --clustering_name ${clustering_name} \
                --clustering_method ${clustering_method}
        done
    done
done
