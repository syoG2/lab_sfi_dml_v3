#!/bin/bash

source_dir=../../source/verb_clustering
data_dir=../../data/verb_clustering
input_dir=../../data/preprocessing_framenet

setting_prefix=all
n_splits=3

python ${source_dir}/make_dataset.py \
    --input_file ${input_dir}/stanza/exemplars.jsonl \
    --output_dir ${data_dir}/dataset \
    --setting_prefix ${setting_prefix} \
    --n_splits ${n_splits}
