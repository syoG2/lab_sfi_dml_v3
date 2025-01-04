#!/bin/bash

source_dir=./source/preprocessing
data_dir=./data/preprocessing/c4/collect

split_name="train"


for ((file_id=0; file_id<10; file_id++));
do
formatted_file_id=$(printf "%05d" ${file_id})
uv run python ${source_dir}/collect_c4.py\
    --file_id ${file_id} \
    --split_name ${split_name} \
    --output_file "${data_dir}/${split_name}_${formatted_file_id}.jsonl"
done