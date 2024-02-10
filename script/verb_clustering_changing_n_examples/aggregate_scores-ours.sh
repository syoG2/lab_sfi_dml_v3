#!/bin/bash

source_dir=../../source/verb_clustering
data_dir=../../data/verb_clustering_changing_n_examples

setting_prefixes=(vf01 vf02 vf05 vf10 vf20)
setting_suffixes=(3_0 3_1 3_2)

pretrained_model_name=bert-base-uncased
# pretrained_model_name=roberta-base
# pretrained_model_name=bert-large-uncased
# pretrained_model_name=roberta-large

model_names=(vanilla softmax_classification adacos_classification siamese_distance triplet_distance arcface_classification)

vec_types=(wm)

clustering_name_methods=(onestep-average twostep-xmeans-average)

split=test

for setting_prefix in ${setting_prefixes[@]}; do
    for model_name in ${model_names[@]}; do
        for vec_type in ${vec_types[@]}; do
            for clustering_name_method in ${clustering_name_methods[@]}; do
                d2=${pretrained_model_name}/${model_name}
                d3=${vec_type}/${clustering_name_method}

                input_dirs=()
                for setting_suffix in ${setting_suffixes[@]}; do
                    d1=${setting_prefix}_${setting_suffix}
                    input_dirs+=(${data_dir}/evaluate_clustering_ours/${d1}/${d2}/${d3})
                done

                d4=${setting_prefix}
                python ${source_dir}/aggregate_scores.py \
                    --input_dirs ${input_dirs[@]} \
                    --output_dir ${data_dir}/aggregate_scores_clustering/${d4}/${d2}/${d3} \
                    --split ${split}
            done
        done
    done
done
