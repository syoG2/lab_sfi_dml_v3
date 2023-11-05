#!/bin/bash

source_dir=../../source/verb_clustering
data_dir=../../data/verb_clustering

settings=(all_3_0 all_3_1 all_3_2)

pretrained_model_name=bert-base-uncased

vec_types=(word mask)

model_name=vanilla

run_numbers=(00)

device=cuda:0

for setting in ${settings[@]}; do
    for vec_type in ${vec_types[@]}; do
        for run_number in ${run_numbers[@]}; do
            d1=${setting}
            d2=${pretrained_model_name}/${model_name}/${vec_type}/${run_number}
            python ${source_dir}/train_model.py \
                --input_train_file ${data_dir}/dataset/${d1}/exemplars_train.jsonl \
                --input_dev_file ${data_dir}/dataset/${d1}/exemplars_dev.jsonl \
                --output_dir ${data_dir}/train_model/${d1}/${d2} \
                --pretrained_model_name ${pretrained_model_name} \
                --model_name ${model_name} \
                --vec_type ${vec_type} \
                --run_number ${run_number} \
                --normalization true \
                --device ${device} \
                --batch_size 32
        done
    done
done
