# ほとんどがC4の用例からなるクラスタを調べる
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)
pd.set_option("display.expand_frame_repr", False)

RED = "\033[31m"
END = "\033[0m"


def list_cluster(
    verb_form, add_method, c4_rate, setting, model_name, clustering_method
):
    output_txt = ""
    output_txt += f"verb_form: {verb_form} add_method: {add_method} c4_rate: {c4_rate} setting: {setting} model_name: {model_name} clustering_method: {clustering_method}\n"

    input_file = Path(
        f"./data/verb_clustering_c4/clustering/{verb_form}/{add_method}/{c4_rate}/{setting}/bert-base-uncased/{model_name}/wm/{clustering_method}/exemplars_test.jsonl"
    )
    if not input_file.exists():
        output_txt += "not found\n"
        return output_txt
    df = pd.read_json(input_file, lines=True)
    group = df.groupby(["frame_cluster", "lu_name"]).size()

    # frame_clusterごとにsourceがc4の割合とサイズを計算
    group = df.groupby("frame_cluster")
    c4_ratio = group["source"].apply(lambda x: (x == "c4").mean())
    cluster_size = group.size()
    framenet_size = group["source"].apply(lambda x: (x == "framenet").sum())
    c4_size = group["source"].apply(lambda x: (x == "c4").sum())
    lu_counts = df.groupby("frame_cluster")["lu_name"].value_counts()
    lu_counts = lu_counts.groupby(level=0).apply(
        lambda x: [{key[1]: x[key]} for key in x.keys()]
    )
    # データフレームにまとめる
    c4_ratio_df = pd.DataFrame(
        {
            "c4_ratio": c4_ratio,
            "size": cluster_size,
            "framenet_size": framenet_size,
            "c4_size": c4_size,
            "lu_counts": lu_counts,
        }
    )
    # c4_ratioで降順ソート、同率の場合はsizeで降順ソート
    c4_ratio_sorted = c4_ratio_df.sort_values(
        by=["c4_ratio", "size"], ascending=[False, False]
    )

    output_txt += f"{c4_ratio_sorted}\n\n"

    for i, cluster_id in enumerate(c4_ratio_sorted.index[:5]):
        output_txt += f"verb_form: {verb_form} add_method: {add_method} c4_rate: {c4_rate} setting: {setting} model_name: {model_name} clustering_method: {clustering_method} i: {i} cluster_id: {cluster_id}\n"

        cluster = df[df["frame_cluster"] == cluster_id].copy()
        cluster["text_widx"] = cluster.apply(
            lambda row: " ".join(
                np.insert(
                    row["text_widx"].split(),
                    [row["target_widx"], row["target_widx"] + 1],
                    ["##", "##"],
                )
            ),
            axis=1,
        )
        cluster = cluster.sort_values(by=["source", "lu_name"], ascending=False)
        cluster = cluster[["frame", "lu_name", "text_widx"]]
        output_txt += f"{c4_ratio_sorted['lu_counts'][cluster_id]}\n"
        output_txt += f"{cluster}\n\n"

    return output_txt


verb_forms = ["lemma", "original"]
add_methods = ["ratio", "sequential", "c4first"]
c4_rates = [0, 1, 2]
settings = ["all_3_0", "all_3_1", "all_3_2"]
model_names = [
    "vanilla",
    "softmax_classification",
    "adacos_classification",
    "siamese_distance",
    "triplet_distance",
    "arcface_classification",
]
clustering_methods = [
    "onestep-average",
    "twostep-xmeans-average",
    "twostep_lu-xmeans-average",
]

txt = list_cluster(
    verb_forms[0],
    add_methods[2],
    c4_rates[1],
    settings[2],
    model_names[0],
    clustering_methods[2],
)

with open("./viewdata/cluster.txt", "w") as f:
    f.write(txt)
