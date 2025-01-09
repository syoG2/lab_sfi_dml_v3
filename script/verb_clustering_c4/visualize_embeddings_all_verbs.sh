#!/bin/bash

source_dir=./source/verb_clustering_c4
data_dir=./data/verb_clustering_c4

settings=(all_3_0 all_3_1 all_3_2)
pretrained_model_name=bert-base-uncased
model_names=(vanilla arcface_classification softmax_classification siamese_distance adacos_classification triplet_distance)

vec_type=word
vec_type=mask

# vec_type=one
# vec_type=wm

# for model_name in "${model_names[@]}"; do
#     for setting in "${settings[@]}"; do
#         for random_state in $(seq 0 9); do
#             d1=${setting}/${pretrained_model_name}
#             d2=${model_name}
#             d3=${vec_type}
#             uv run python ${source_dir}/visualize_embeddings_all_verbs.py \
#                 --input_dir "${data_dir}/embeding/${d1}/${d2}" \
#                 --input_params_file "${data_dir}/best_params_ranking/${d1}/${d2}/${d3}/best_params.json" \
#                 --output_dir "${data_dir}/visualization_all_verbs/${d1}/${d2}/${d3}" \
#                 --vec_type ${vec_type} \
#                 --random_state "${random_state}"
#         done
#     done
# done

# vec_type=word
# alpha=0
# for model_name in vanilla triplet_distance adacos_classification; do
#     for setting in all_3_0; do
#         for random_state in 3; do
#             d1=${setting}
#             d2=${model_name}
#             uv run python ${source_dir}/visualize_embeddings_all_verbs.py \
#                 --input_dir ${data_dir}/evaluation${d1}/embeddings \
#                 --input_mlruns_dir ${data_dir}/evaluation/${d1}/mlruns \
#                 --output_dir ${data_dir}/visualization_all_verbs2/${d1}/${d2} \
#                 --setting ${setting} \
#                 --model_name ${model_name} \
#                 --vec_type ${vec_type} \
#                 --random_state ${random_state} \
#                 --alpha ${alpha}
#         done
#     done
# done

clustering_name=twostep
clustering_method1=xmeans
clustering_method2=average

for model_name in "${model_names[@]}"; do
    for setting in "${settings[@]}"; do
        for random_state in $(seq 0 9); do
            d1=${setting}/${pretrained_model_name}
            d2=${model_name}
            d3=${vec_type}/${clustering_name}-${clustering_method1}-${clustering_method2}
            uv run python ${source_dir}/visualize_embeddings_all_verbs.py \
                --input_dir "${data_dir}/embedding/${d1}/${d2}" \
                --input_params_file "${data_dir}/best_params_clustering/${d1}/${d2}/${d3}/best_params.json" \
                --output_dir "${data_dir}/visualization_all_verbs_clustering/${d1}/${d2}/${d3}" \
                --vec_type ${vec_type} \
                --random_state "${random_state}"
        done
    done
done