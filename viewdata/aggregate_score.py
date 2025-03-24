# aggregate_scoreで実行した結果をmarkdownの表形式で出力する
from pathlib import Path

import pandas as pd

# 動詞の取り方2種類
# verb_forms = ["lemma","original"]

# LU抽出モデルの入力方法3種類
# text_input_styles = ["sep","token0","token00"]

# 深層距離学習手法6種類
# model_names = ["vanilla","siamese_distance","triplet_distance","softmax_classification","arcface_classification","adacos_classification"]
# クラスタリング手法3種類
# clusterings = ["onestep-average","twostep-xmeans-average","twostep_lu-xmeans-average"]
# c4の混ぜ方6種類
# add_methods = ["ratio","sequential","c4first","ratio_verb","sequential_verb","c4first_verb"]


def make_table(
    verb_form="",
    add_method="",
    add_key="",
    clustering_dataset="c4first",
    c4_rate=-1,
    input_dir="",
    vec_type="wm",
):
    # スコアをmarkdownの表形式で出力
    output_txt = ""
    if (input_dir == "") and (
        verb_form == ""
        or add_method == ""
        or add_key == ""
        or clustering_dataset == ""
        or c4_rate < 0
    ):
        output_txt += f"{verb_form},{add_method},{add_key},{clustering_dataset},{c4_rate},{vec_type}のいずれかが指定されていません\n"

        return output_txt

    if input_dir != "":
        output_txt += f"{input_dir}\n"
    else:
        output_txt += f"{verb_form}/{add_method}/{add_key}/{clustering_dataset}/{c4_rate}/{vec_type}\n"

    df = pd.DataFrame()
    key = [
        "ave-alpha",
        # "ave-n_pred_lus_framenet",
        "ave-n_pred_lus_c4",
        "ave-n_pred_clusters",
        "ave-n_pred_clusters_c4",
        "ave-pu",
        "ave-ipu",
        "ave-puf",
        "ave-bcp",
        "ave-bcr",
        "ave-bcf",
    ]
    model_names = [
        "vanilla",
        "siamese_distance",
        "triplet_distance",
        "softmax_classification",
        "arcface_classification",
        "adacos_classification",
    ]
    clusterings = [
        "onestep-average",
        "twostep-xmeans-average",
        "twostep_lu-xmeans-average",
    ]
    for clustering in clusterings:
        for model_name in model_names:
            if input_dir != "":
                input_file = Path(
                    input_dir + f"/{model_name}/wm/{clustering}/metrics_test.json"
                )
            else:
                input_file = Path(
                    f"./data/verb_clustering_c4/aggregate_scores_clustering/{verb_form}/{add_method}/{add_key}/{clustering_dataset}/{c4_rate}/bert-base-uncased/{model_name}/{vec_type}/{clustering}/metrics_test.json"
                )
            series = pd.Series()
            if input_file.exists():
                series = pd.read_json(input_file, typ="series", orient="records")
                df = df.reindex(df.index.union(series.index))
            if clustering == "onestep-average":
                df["1_" + model_name] = series
            elif clustering == "twostep-xmeans-average":
                df["2_" + model_name] = series
            else:
                # df["lu_" + model_name] = series
                continue
    df = df.T
    df = df.reindex(columns=key)
    df[["ave-pu", "ave-ipu", "ave-puf", "ave-bcp", "ave-bcr", "ave-bcf"]] = (
        df[["ave-pu", "ave-ipu", "ave-puf", "ave-bcp", "ave-bcr", "ave-bcf"]] * 100
    ).round(1)
    output_txt += df.to_markdown() + "\n\n"
    output_txt += df.to_latex(float_format="%.1f") + "\n\n"

    return output_txt


def main():
    with open("./viewdata/aggregate_score.txt", "w") as f:
        # # 山田さんの再試結果
        # txt = make_table(
        #     input_dir="./data/verb_clustering/aggregate_scores_clustering/bert-base-uncased/"
        # )
        # f.write(txt + "\n\n")
        # c4を先にクラスタリングした場合の結果
        # txt = make_table(verb_form="lemma", add_method="c4first", c4_rate=1)
        # f.write(txt + "\n")
        # txt = make_table(
        #     verb_form="original",
        #     add_method="c4first",
        #     add_key="verb",
        #     clustering_dataset="c4first",
        #     c4_rate=1,
        # )
        # f.write(txt + "\n\n")
        # txt = make_table(
        #     verb_form="original",
        #     add_method="c4first_verb",
        #     add_key="verb",
        #     clustering_dataset="c4first",
        #     c4_rate=1,
        # )
        # f.write(txt + "\n\n")

        # c4を混ぜてクラスタリングした結果
        for add_method in ["ratio"]:
            for c4_rate in [1]:
                for verb_form in ["original"]:
                    # for vec_type in ["mask", "wm", "word"]:
                    for vec_type in ["wm"]:
                        txt = make_table(
                            verb_form=verb_form,
                            add_method=add_method,
                            add_key="verb",
                            clustering_dataset="c4first",
                            c4_rate=c4_rate,
                            vec_type=vec_type,
                        )
                        f.write(txt + "\n")
                    f.write("\n\n")


if __name__ == "__main__":
    main()
