#!/bin/bash

source_dir=../../source/verb_clustering
data_dir=../../data/verb_clustering_changing_n_examples

# settings=(vf01_3_0 vf01_3_1 vf01_3_2)
# settings=(vf02_3_0 vf02_3_1 vf02_3_2)
# settings=(vf05_3_0 vf05_3_1 vf05_3_2)
# settings=(vf10_3_0 vf10_3_1 vf10_3_2)
settings=(vf20_3_0 vf20_3_1 vf20_3_2)

pretrained_model_name=bert-base-uncased

# model_names=(vanilla softmax_classification adacos_classification)
# run_numbers=(00)

# model_names=(siamese_distance triplet_distance arcface_classification)
# model_names=(siamese_distance)
model_names=(triplet_distance)
# model_names=(arcface_classification)
run_numbers=(00 01 02 03)

# vec_types=(word mask wm)
# vec_types=(word mask)
vec_types=(wm)

clustering_name=onestep
clustering_method=average

for setting in ${settings[@]}; do
    for model_name in ${model_names[@]}; do
        for vec_type in ${vec_types[@]}; do
            d1=${setting}/${pretrained_model_name}/${model_name}
            d2=${vec_type}/${clustering_name}-${clustering_method}
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

clustering_name_method=${clustering_name}-${clustering_method}
splits=(dev test)
for setting in ${settings[@]}; do
    for model_name in ${model_names[@]}; do
        for vec_type in ${vec_types[@]}; do
            for split in ${splits[@]}; do
                d1=${setting}/${pretrained_model_name}/${model_name}
                d2=${vec_type}/${clustering_name_method}
                python ${source_dir}/evaluate_clustering.py \
                    --input_file ${data_dir}/clustering/${d1}/${d2}/exemplars_${split}.jsonl \
                    --input_params_file ${data_dir}/clustering/${d1}/${d2}/params.json \
                    --output_dir ${data_dir}/evaluate_clustering_ours/${d1}/${d2} \
                    --split ${split}
            done
        done
    done
done
