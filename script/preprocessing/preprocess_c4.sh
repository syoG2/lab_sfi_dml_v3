#!/bin/bash
source_dir=./source/preprocessing
data_dir=./data/preprocessing/c4/collect
text_input_style="token0"


output_dir=./data/preprocessing/c4/preprocess/${text_input_style}
model_dir=./data/preprocessing/lu_classifier/models/bert-base-uncased/${text_input_style}/train/5_0

split_name="train"


for ((file_id=5; file_id<6; file_id++));do
    for ((part_id=270; part_id<360; part_id += 1));do
        formatted_file_id=$(printf "%05d" ${file_id})
        uv run python ${source_dir}/preprocess_c4.py\
            --input_file "${data_dir}/${split_name}_${formatted_file_id}.jsonl" \
            --file_id ${file_id} \
            --part_id ${part_id} \
            --split_name ${split_name} \
            --output_exemplar_dir "${output_dir}/${split_name}_${formatted_file_id}"/lu\
            --output_wordlist_file "${output_dir}/${split_name}_${formatted_file_id}/word_list/word_list_${part_id}.jsonl"\
            --device "cuda:0" \
            --model_path "${model_dir}/best_model"\
            --tokenizer_path "${model_dir}/tokenizer"\
            --text_input_style ${text_input_style}
    done
done