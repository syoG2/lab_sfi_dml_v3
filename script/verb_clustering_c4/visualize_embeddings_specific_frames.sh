#!/bin/bash

source_dir=./source/verb_clustering_c4
data_dir=./data/verb_clustering_c4

setting=all_3_1
pretrained_model_name=bert-base-uncased

#model_name=vanilla
model_name=adacos_classification

vec_type=word
alpha=0

frames=(Filling Placing Removing Topic)

for random_state in $(seq 600 699); do
    d1=${setting}/${pretrained_model_name}
    d2=${model_name}
    d3=${vec_type}
    echo "${random_state}"
    uv run python ${source_dir}/visualize_embeddings_specific_frames.py \
        --input_dir ${data_dir}/embeding/${d1}/${d2} \
        --input_params_file ${data_dir}/best_params_ranking/${d1}/${d2}/${d3}/best_params.json \
        --output_dir ${data_dir}/visualization_specific_frames/${d1}/${d2}/${d3} \
        --frames "${frames[@]}" \
        --random_state "${random_state}"
done

#! for EACL papers
#random_state=3 #bert
# random_state=40 #adacos
# echo ${random_state}
# python ../experiment/visualize_embeddings_specific_frames.py \
#     --input_dir ${data_dir}/evaluation${d1}/embeddings \
#     --input_mlruns_dir ${data_dir}/evaluation/${d1}/mlruns \
#     --output_dir ${data_dir}/visualization_specific_frames2/${d1}/${d2} \
#     --setting ${setting} \
#     --model ${model} \
#     --vec_type ${vec_type} \
#     --random_state ${random_state} \
#     --alpha ${alpha}
