#!/bin/bash

source_dir=./source/preprocessing
data_dir=./data/preprocessing/framenet

uv run python ${source_dir}/preprocess_framenet.py\
    --input_file ${data_dir}/collect/exemplars.jsonl\
    --output_exemplar_file ${data_dir}/preprocess/exemplars.jsonl\
    --output_wordlist_file ${data_dir}/preprocess/word_list.jsonl\
    --device cuda:0