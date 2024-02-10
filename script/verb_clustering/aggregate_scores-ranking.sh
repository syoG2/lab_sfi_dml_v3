#!/bin/bash

source_dir=../../source/verb_clustering
data_dir=../../data/verb_clustering

# ranking=ranking
ranking=ranking_overlap

settings=(all_3_0 all_3_1 all_3_2)

pretrained_model_name=bert-base-uncased
# pretrained_model_name=roberta-base
# pretrained_model_name=bert-large-uncased
# pretrained_model_name=roberta-large

model_names=(vanilla softmax_classification adacos_classification siamese_distance triplet_distance arcface_classification)

vec_types=(wm-onestep-average)

ranking_methods=(all-all same-same diff-diff)

split=test

for model_name in ${model_names[@]}; do
    for vec_type in ${vec_types[@]}; do
        for ranking_method in ${ranking_methods[@]}; do
            d2=${pretrained_model_name}/${model_name}
            d3=${vec_type}/${ranking_method}

            input_dirs=()
            for setting in ${settings[@]}; do
                d1=${setting}
                input_dirs+=(${data_dir}/${ranking}/${d1}/${d2}/${d3})
            done

            python ${source_dir}/aggregate_scores.py \
                --input_dirs ${input_dirs[@]} \
                --output_dir ${data_dir}/aggregate_scores_${ranking}/${d2}/${d3} \
                --split ${split}
        done
    done
done
