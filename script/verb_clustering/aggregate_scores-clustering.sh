#!/bin/bash

source_dir=../../source/verb_clustering
data_dir=../../data/verb_clustering

settings=(all_3_0 all_3_1 all_3_2)

pretrained_model_name=bert-base-uncased
model_names=(vanilla softmax_classification adacos_classification siamese_distance triplet_distance arcface_classification)

vec_types=(word mask wm)

clustering_name_methods=(onestep-average twostep-xmeans-average)

split=test

for model_name in ${model_names[@]}; do
    for vec_type in ${vec_types[@]}; do
        for clustering_name_method in ${clustering_name_methods[@]}; do
            d2=${pretrained_model_name}/${model_name}
            d3=${vec_type}/${clustering_name_method}

            input_files=()
            for setting in ${settings[@]}; do
                d1=${setting}
                input_files+=(${data_dir}/evaluate_clustering/${d1}/${d2}/${d3}/metrics_${split}.json)
            done

            python ${source_dir}/aggregate_scores.py \
                --input_files ${input_files[@]} \
                --output_dir ${data_dir}/aggregate_scores_clustering/${d2}/${d3} \
                --split ${split}
        done
    done
done
