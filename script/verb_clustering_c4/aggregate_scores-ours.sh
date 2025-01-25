#!/bin/bash

source_dir=./source/verb_clustering_c4
data_dir=./data/verb_clustering_c4

settings=(all_3_0 all_3_1 all_3_2)

pretrained_model_name=bert-base-uncased
# pretrained_model_name=roberta-base
# pretrained_model_name=bert-large-uncased
# pretrained_model_name=roberta-large

model_names=(vanilla softmax_classification adacos_classification siamese_distance triplet_distance arcface_classification)

vec_types=(mask word wm)
# vec_types=(wm)

clustering_name_methods=(onestep-average twostep-xmeans-average twostep_lu-xmeans-average)

split="test"

add_methods=(ratio sequential c4first c4first_verb)
c4_rates=(0 1 2)

verb_forms=(original lemma)

for verb_form in "${verb_forms[@]}"; do
    for add_method in "${add_methods[@]}"; do
        for c4_rate in "${c4_rates[@]}"; do
            for model_name in "${model_names[@]}"; do
                for vec_type in "${vec_types[@]}"; do
                    for clustering_name_method in "${clustering_name_methods[@]}"; do
                        d2=${pretrained_model_name}/${model_name}
                        d3=${vec_type}/${clustering_name_method}

                        input_dirs=()
                        for setting in "${settings[@]}"; do
                            d1=${setting}
                            input_dirs+=("${data_dir}/evaluate_clustering_ours/${verb_form}/${add_method}/${c4_rate}/${d1}/${d2}/${d3}")
                        done

                        uv run python ${source_dir}/aggregate_scores.py \
                            --input_dirs "${input_dirs[@]}" \
                            --output_dir "${data_dir}/aggregate_scores_clustering/${verb_form}/${add_method}/${c4_rate}/${d2}/${d3}" \
                            --split ${split}
                    done
                done
            done
        done
    done
done