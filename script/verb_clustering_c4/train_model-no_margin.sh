#!/bin/bash

source_dir=./source/verb_clustering_c4
data_dir=./data/verb_clustering_c4

settings=(all_3_0 all_3_1 all_3_2)

pretrained_model_name=bert-base-uncased

# vec_types=(word mask)
# vec_types=(word)
vec_types=(mask)

# model_name=softmax_classification
model_name=adacos_classification

run_numbers=(00)

device=cuda:3

# c4_rateの値はどれでも良い
c4_rate=0

# add_methodはどちらでも同じ
# add_method=sequential
add_method=ratio

for setting in "${settings[@]}"; do
    for vec_type in "${vec_types[@]}"; do
        for run_number in "${run_numbers[@]}"; do
            d1=${setting}
            d2=${pretrained_model_name}/${model_name}/${vec_type}/${run_number}
            uv run python ${source_dir}/train_model.py \
                --input_train_file "${data_dir}/dataset/${add_method}/${c4_rate}/${d1}/exemplars_train.jsonl" \
                --input_dev_file "${data_dir}/dataset/${add_method}/${c4_rate}/${d1}/exemplars_dev.jsonl" \
                --output_dir "${data_dir}/train_model/${d1}/${d2}" \
                --pretrained_model_name ${pretrained_model_name} \
                --model_name ${model_name} \
                --vec_type "${vec_type}" \
                --run_number "${run_number}" \
                --normalization true \
                --device ${device} \
                --batch_size 32 \
                --learning_rate 1e-5 \
                --n_epochs 5
        done
    done
done
