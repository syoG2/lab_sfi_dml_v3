#!/bin/bash

source_dir=../../source/verb_clustering
data_dir=../../data/verb_clustering

# settings=(all_3_0 all_3_1 all_3_2)
settings=(all_3_2)

device=cuda:3
for setting in ${settings[@]}; do
    d1=${setting}
    python ${source_dir}/perform_previous_clustering_arefyev.py \
        --input_dev_file ${data_dir}/dataset/${d1}/exemplars_dev.jsonl \
        --input_test_file ${data_dir}/dataset/${d1}/exemplars_test.jsonl \
        --output_dir ${data_dir}/previous_clustering_arefyev/${d1} \
        --batch_size 32 \
        --device ${device}
done
