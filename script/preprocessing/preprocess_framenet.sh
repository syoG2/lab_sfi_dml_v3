#!/bin/bash

source_dir=./source/preprocessing
data_dir=./data/preprocessing/framenet

verb_form=original
# verb_form=lemma
uv run python ${source_dir}/preprocess_framenet.py \
    --input_file ${data_dir}/collect/exemplars.jsonl \
    --output_exemplar_file ${data_dir}/preprocess/${verb_form}/exemplars.jsonl \
    --output_wordlist_file ${data_dir}/preprocess/${verb_form}/word_list.jsonl \
    --device cuda:0 \
    --verb_form ${verb_form}