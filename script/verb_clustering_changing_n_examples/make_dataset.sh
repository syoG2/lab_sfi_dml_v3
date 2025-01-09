#!/bin/bash

source_dir=./source/verb_clustering_changing_n_examples
data_dir=./data/verb_clustering_changing_n_examples
input_dir=./data/preprocessing_framenet

setting_prefix_list=(vf01 vf02 vf05 vf10 vf20)
n_splits=3

for setting_prefix in "${setting_prefix_list[@]}"; do
    uv run python ${source_dir}/make_dataset.py \
        --input_file ${input_dir}/stanza/exemplars.jsonl \
        --output_dir ${data_dir}/dataset \
        --setting_prefix "${setting_prefix}" \
        --n_splits ${n_splits}
done
