#!/bin/bash

source_dir=../../source/verb_clustering
data_dir=../../data/verb_clustering

# settings=(all_3_0 all_3_1 all_3_2)
settings=(all_3_2)

pretrained_model_name=bert-base-uncased

model_names=(vanilla arcface_classification softmax_classification siamese_distance adacos_classification triplet_distance)

vec_type=onestep-average
clustering_name=onestep
clustering_method=average

vec_type2=wm

ranking_methods=(all-all same-same diff-diff)

split=test

for setting in ${settings[@]}; do
    for model_name in ${model_names[@]}; do
        for ranking_method in ${ranking_methods[@]}; do
            d1=${setting}/${pretrained_model_name}/${model_name}
            d2=${vec_type2}/${clustering_name}/${clustering_method}
            d3=${vec_type}
            python ${source_dir}/evaluate_ranking.py \
                --input_dir ${data_dir}/embedding/${d1} \
                --input_params_file ${data_dir}/best_params_clustering/${d1}/${d2}/best_params.json \
                --output_dir ${data_dir}/ranking/${d1}/${d3} \
                --ranking_method ${ranking_method} \
                --split ${split}

            python ${source_dir}/evaluate_ranking_overlap.py \
                --input_score_file ${data_dir}/ranking/${d1}/${d3}/score_${split}.jsonl \
                --input_train_file ${data_dir}/embedding/${d1}/exemplars_train.jsonl \
                --output_dir ${data_dir}/ranking_overlap/${d1}/${d3} \
                --split ${split}
        done
    done
done
