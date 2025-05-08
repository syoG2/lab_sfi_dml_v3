#!/bin/bash

source_dir=../../source/verb_clustering
data_dir=../../data/verb_clustering_changing_n_examples

settings=(vf01_3_0 vf01_3_1 vf01_3_2)
settings+=(vf02_3_0 vf02_3_1 vf02_3_2)
settings+=(vf05_3_0 vf05_3_1 vf05_3_2)
settings+=(vf10_3_0 vf10_3_1 vf10_3_2)
settings+=(vf20_3_0 vf20_3_1 vf20_3_2)

pretrained_model_name=bert-base-uncased

# vec_types=(word mask)
# vec_types=(word)
vec_types=(mask)

model_name=vanilla

run_numbers=(00)

device=cuda:1

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

splits=(train dev test)
for setting in ${settings[@]}; do
    for vec_type in ${vec_types[@]}; do
        for run_number in ${run_numbers[@]}; do
            for split in ${splits[@]}; do
                d1=${setting}
                d2=${pretrained_model_name}/${model_name}/${vec_type}/${run_number}
                python ${source_dir}/get_embedding.py \
                    --input_file ${data_dir}/dataset/${d1}/exemplars_${split}.jsonl \
                    --input_params_file ${data_dir}/train_model/${d1}/${d2}/params.json \
                    --input_model_file ${data_dir}/train_model/${d1}/${d2}/pretrained_model_last.pth \
                    --output_dir ${data_dir}/embedding/${d1}/${d2}
            done
        done
    done
done
