#!/bin/bash
# [ ]:C4で先にクラスタリングを行い、そのクラスタの最近傍点にFrameNetの用例を割り当てるスクリプトを作成

source_dir=./source/verb_clustering_c4
data_dir=./data/verb_clustering_c4

settings=(all_3_0 all_3_1 all_3_2)

pretrained_model_name=bert-base-uncased
# pretrained_model_name=bert-large-uncased
# pretrained_model_name=roberta-base
# pretrained_model_name=roberta-large

# model_names=(vanilla softmax_classification adacos_classification)
# model_names=(vanilla)
# model_names=(softmax_classification)
# model_names=(adacos_classification)
# run_numbers=(00)

# model_names=(siamese_distance triplet_distance arcface_classification)
# model_names=(arcface_classification)
# model_names=(siamese_distance)
model_names=(triplet_distance)
run_numbers=(00 01 02 03)

vec_types=(mask wm word)
# vec_types=(mask)
# vec_types=(wm)
# vec_types=(word)

clustering_name=twostep
clustering_method1=xmeans
clustering_method2=average

c4_rate=2

# add_method=ratio
add_method=sequential

for setting in "${settings[@]}"; do
    for model_name in "${model_names[@]}"; do
        for vec_type in "${vec_types[@]}"; do
            d1=${setting}/${pretrained_model_name}/${model_name}
            d2=${vec_type}/${clustering_name}-${clustering_method1}-${clustering_method2}

            uv run python ${source_dir}/find_best_params_clustering.py \
                --input_dir "${data_dir}/embedding/${add_method}/${c4_rate}/${d1}" \
                --output_dir "${data_dir}/best_params_clustering/${add_method}/${c4_rate}/${d1}/${d2}" \
                --vec_type "${vec_type}" \
                --run_numbers "${run_numbers[@]}" \
                --clustering_name ${clustering_name} \
                --clustering_method1 ${clustering_method1} \
                --clustering_method2 ${clustering_method2}

            # [ ]:C4のみでクラスタリングするようにファイルを入力
            uv run python ${source_dir}/perform_clustering.py \
                --input_dir "${data_dir}/embedding/${add_method}/${c4_rate}/${d1}" \
                --output_dir "${data_dir}/clustering/${add_method}/${c4_rate}/${d1}/${d2}" \
                --input_params_file "${data_dir}/best_params_clustering/${add_method}/${c4_rate}/${d1}/${d2}/best_params.json" \
                --clustering_name ${clustering_name} \
                --clustering_method1 ${clustering_method1} \
                --clustering_method2 ${clustering_method2}

            # [ ]:C4でクラスタリングしたものにFrameNetの用例を割り当てるスクリプトを作成

            splits=(dev test)
            for split in "${splits[@]}"; do
                uv run python ${source_dir}/evaluate_clustering.py \
                    --input_file "${data_dir}/clustering/${add_method}/${c4_rate}/${d1}/${d2}/exemplars_${split}.jsonl" \
                    --input_params_file "${data_dir}/clustering/${add_method}/${c4_rate}/${d1}/${d2}/params.json" \
                    --output_dir "${data_dir}/evaluate_clustering_ours/${add_method}/${c4_rate}/${d1}/${d2}" \
                    --split "${split}"
            done
        done
    done
done
