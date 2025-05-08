#!/bin/bash

source_dir=./source/verb_clustering_changing_n_examples_c4
data_dir=./data/verb_clustering_changing_n_examples_c4
input_dir=./data/preprocessing/framenet/preprocess

setting_prefix_list=(vf01 vf02 vf05 vf10 vf20)
n_splits=3

# verb_form: 2段階クラスタリングにおける、1段階目の区分
#   originalでは動詞の原型、lemmaではstanzaにおけるlemmaをもとに区分
# verb_form=lemma
verb_form=original

# add_method: C4のデータの追加方法
#   frequency_100: 用例の最大数を100として、現れた順に用例を追加する
#   ratio: 各用例の出現頻度に基づいて、用例を追加する
add_method=frequency_100
add_method=ratio

# add_key: FrameNetの用例の何を基準にしてC4のデータを追加するか
#   lu_name: FrameNetの用例のlu_nameを基準にする
#   verb: FrameNetの用例の動詞を基準にする
# add_key=lu_name
add_key=verb

# clustering_dataset: クラスタリングの対象とするデータセット
#   c4first: C4のデータを最初にクラスタリングしてから、FrameNetのデータを割り当てる
#   mix: C4のデータとFrameNetのデータを混ぜてクラスタリングする
# clustering_dataset=c4first
# clustering_dataset=mix

# c4_rate: FrameNetの用例に対し、C4のデータを追加する割合
c4_rate=2

# c4_file_id: C4のデータのファイルID(0~c4_file_idまでのファイルを読み込む)
c4_file_id=6

for setting_prefix in "${setting_prefix_list[@]}"; do
    uv run python ${source_dir}/make_dataset.py \
        --input_file ${input_dir}/stanza/exemplars.jsonl \
        --output_dir ${data_dir}/dataset \
        --setting_prefix "${setting_prefix}" \
        --n_splits ${n_splits}
done
