#!/bin/bash

source_dir=../../source/verb_clustering
data_dir=../../data/verb_clustering

# settings=(all_3_0 all_3_1 all_3_2)
settings=(all_3_1)

pretrained_model_name=bert-base-uncased

# model_names=(vanilla softmax_classification adacos_classification)
# run_numbers=(00)
# model_names=(siamese_distance triplet_distance arcface_classification)
# run_numbers=(00 01 02 03)
model_names=(triplet_distance)
run_numbers=(00 01 02 03)

# vec_types=(word mask wm)
vec_types=(word)
# vec_types=(mask)
# vec_types=(wm)

clustering_name=twostep
clustering_method1=xmeans
clustering_method2=average

for setting in ${settings[@]}; do
    for model_name in ${model_names[@]}; do
        for vec_type in ${vec_types[@]}; do
            d1=${setting}/${pretrained_model_name}/${model_name}
            d2=${vec_type}/${clustering_name}-${clustering_method1}-${clustering_method2}
            python ${source_dir}/find_best_params_clustering.py \
                --input_dir ${data_dir}/embedding/${d1} \
                --output_dir ${data_dir}/best_params_clustering/${d1}/${d2} \
                --vec_type ${vec_type} \
                --run_numbers ${run_numbers[@]} \
                --clustering_name ${clustering_name} \
                --clustering_method1 ${clustering_method1} \
                --clustering_method2 ${clustering_method2}
        done
    done
done
