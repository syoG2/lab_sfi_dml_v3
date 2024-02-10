#!/bin/bash

source_dir=../../source/verb_clustering
data_dir=../../data/verb_clustering

settings=(all_3_0 all_3_1 all_3_2)

clustering_name_methods=(arefyev anwar ribeiro)

split=test

for clustering_name_method in ${clustering_name_methods[@]}; do
    d2=${clustering_name_method}

    input_dirs=()
    for setting in ${settings[@]}; do
        d1=${setting}
        input_dirs+=(${data_dir}/evaluate_clustering_previous/${d1}/${d2})
    done

    python ${source_dir}/aggregate_scores.py \
        --input_dirs ${input_dirs[@]} \
        --output_dir ${data_dir}/aggregate_scores_clustering/${d2} \
        --split ${split}
done
