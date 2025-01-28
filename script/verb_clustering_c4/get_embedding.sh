#!/bin/bash

source_dir=./source/verb_clustering_c4
data_dir=./data/verb_clustering_c4

settings=(all_3_0 all_3_1 all_3_2)
# settings=(all_3_0)

pretrained_model_name=bert-base-uncased

vec_types=(mask word)
# vec_types=(mask)
# vec_types=(word)

# model_names=(adacos_classification softmax_classification vanilla)
# model_names=(adacos_classification)
# model_names=(softmax_classification)
model_names=(vanilla)
run_numbers=(00)

# model_names=(arcface_classification siamese_distance triplet_distance)
# model_names=(arcface_classification)
# model_names=(siamese_distance)
# model_names=(triplet_distance)
# run_numbers=(00 01 02 03)

# splits=(train dev test)
# splits=(train)
# splits=(dev)
# splits=(test)

# add_method=ratio
# add_method=sequential
# add_method=ratio_verb

add_method=sequential_n_verb

c4_rate=1

device=cuda:0


add_method=c4first
# add_method=c4first_verb
c4_rate=1
splits=(train dev test-c4 test-framenet) # add_method=c4firstの場合
# splits=(train)
# splits=(dev)
# splits=(test-c4)
# splits=(test-framenet)

verb_form=original
# verb_form=lemma

for setting in "${settings[@]}"; do
    for model_name in "${model_names[@]}"; do
        for vec_type in "${vec_types[@]}"; do
            for run_number in "${run_numbers[@]}"; do
                for split in "${splits[@]}"; do
                    d1=${setting}
                    d2=${pretrained_model_name}/${model_name}/${vec_type}/${run_number}
                    uv run python ${source_dir}/get_embedding.py \
                        --input_file "${data_dir}/dataset/${verb_form}/${add_method}/${c4_rate}/${d1}/exemplars_${split}.jsonl" \
                        --input_params_file "${data_dir}/train_model/${verb_form}/${d1}/${d2}/params.json" \
                        --input_model_file "${data_dir}/train_model/${verb_form}/${d1}/${d2}/pretrained_model_last.pth" \
                        --output_dir "${data_dir}/embedding/${verb_form}/${add_method}/${c4_rate}/${d1}/${d2}" \
                        --device ${device}
                done
            done
        done
    done
done
