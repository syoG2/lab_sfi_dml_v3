# ほとんどがC4の用例からなるクラスタを調べる
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from sfidml.f_induc.embedding import read_embedding
from sfidml.utils.data_utils import read_json, read_jsonl

pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)
pd.set_option("display.expand_frame_repr", False)

RED = "\033[31m"
END = "\033[0m"


def list_cluster(
    verb_form,
    add_method,
    add_key,
    clustering_dataset,
    c4_rate,
    settings,
    model_name,
    clustering_method,
    vec_type="wm",
    top_n=5,
):
    output_txt = ""
    output_txt += f"verb_form: {verb_form} add_method: {add_method} add_key: {add_key} clustering_dataset: {clustering_dataset} c4_rate: {c4_rate} settings: {settings} model_name: {model_name} clustering_method: {clustering_method} vec_type: {vec_type}\n"
    all_c4_ratio_df = pd.DataFrame()
    all_df = pd.DataFrame()
    for setting in settings:
        input_file = Path(
            f"./data/verb_clustering_c4/clustering/{verb_form}/{add_method}/{add_key}/{clustering_dataset}/{c4_rate}/{setting}/bert-base-uncased/{model_name}/{vec_type}/{clustering_method}/exemplars_test.jsonl"
        )
        if not input_file.exists():
            output_txt += "not found\n"
            return output_txt
        df = pd.read_json(input_file, lines=True)
        df["setting"] = setting
        all_df = pd.concat([all_df, df])
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
        # importance = group["source"].apply(lambda x: (x == "c4").mean() * (x == "c4").sum())

        # データフレームにまとめる
        c4_ratio_df = pd.DataFrame(
            {
                # "importance": importance,
                "setting": setting,
                "c4_ratio": c4_ratio,
                "size": cluster_size,
                "framenet_size": framenet_size,
                "c4_size": c4_size,
                "lu_counts": lu_counts,
            }
        )
        all_c4_ratio_df = pd.concat([all_c4_ratio_df, c4_ratio_df])

    by = ["c4_ratio", "c4_size"]
    # c4_ratioで降順ソート、同率の場合はsizeで降順ソート
    c4_ratio_sorted = all_c4_ratio_df.sort_values(by=by, ascending=[False] * len(by))

    # output_txt += f"{c4_ratio_sorted[:top_n].to_markdown()}\n\n"
    output_txt += f"{c4_ratio_sorted.to_markdown()}\n\n"

    # for i, (cluster_id, row) in enumerate(c4_ratio_sorted.iloc[:top_n].iterrows()):
    for i, (cluster_id, row) in enumerate(c4_ratio_sorted.iterrows()):
        setting = row["setting"]
        output_txt += f"{i} verb_form: {verb_form} add_method: {add_method} c4_rate: {c4_rate} setting: {setting} model_name: {model_name} clustering_method: {clustering_method} vec_type:{vec_type} i: {i} cluster_id: {cluster_id}\n"

        cluster = all_df[
            (all_df["frame_cluster"] == cluster_id) & (all_df["setting"] == setting)
        ].copy()
        cluster["text_widx"] = cluster.apply(
            lambda row: " ".join(
                np.insert(
                    row["text_widx"].split(),
                    [row["target_widx"], row["target_widx"] + 1],
                    ["**", "**"],
                )
            ),
            axis=1,
        )
        cluster = cluster.sort_values(by=["source", "lu_name"], ascending=False)
        cluster = cluster[["frame", "lu_name", "text_widx"]]
        output_txt += f"{c4_ratio_sorted[c4_ratio_sorted['setting'] == setting]['lu_counts'][cluster_id]}\n"
        output_txt += f"{cluster}\n\n"

    return output_txt


def get_distinct_clusters(
    verb_form, add_method, c4_rate, setting, model_name, clustering_method, top_n=5
):
    output_txt = ""
    output_txt += f"verb_form: {verb_form} add_method: {add_method} c4_rate: {c4_rate} setting: {setting} model_name: {model_name} clustering_method: {clustering_method}\n"

    args = argparse.ArgumentParser().parse_args()
    args.input_params_file = Path(
        f"./data/verb_clustering_c4/best_params_clustering/{verb_form}/{add_method}/{c4_rate}/{setting}/bert-base-uncased/{model_name}/wm/{clustering_method}/best_params.json"
    )
    if args.input_params_file.exists():
        best_params = read_json(args.input_params_file)
        for key, value in best_params.items():
            setattr(args, key, value)
    else:
        output_txt += "not found\n"
        return output_txt

    args.input_clustering_dir = Path(
        f"./data/verb_clustering_c4/clustering/{verb_form}/{add_method}/{c4_rate}/{setting}/bert-base-uncased/{model_name}/wm/{clustering_method}"
    )
    c4_cluster = pd.DataFrame(
        read_jsonl(args.input_clustering_dir / "exemplars_test-c4.jsonl")
    )

    args.input_embedding_dir = Path(
        f"./data/verb_clustering_c4/embedding/{verb_form}/{add_method}/{c4_rate}/{setting}/bert-base-uncased/{model_name}"
    )

    df_vec_c4, vec_array_c4 = read_embedding(
        args.input_embedding_dir, "test-c4", args.vec_type2run_number, args.alpha
    )
    df_vec_c4 = pd.merge(df_vec_c4, c4_cluster, on=["ex_idx", "vec_id"])

    df_vec_framenet, vec_array_framenet = read_embedding(
        args.input_embedding_dir, "test-framenet", args.vec_type2run_number, args.alpha
    )
    extractor = NearestNeighbors(metric="euclidean", n_neighbors=1)
    extractor.fit(vec_array_framenet)

    all_cluster = pd.DataFrame(
        read_jsonl(args.input_clustering_dir / "exemplars_test.jsonl")
    )

    for cluster_id in tqdm(df_vec_c4["frame_cluster"].unique()):
        mean = np.mean(
            vec_array_c4[df_vec_c4[df_vec_c4["frame_cluster"] == cluster_id]["vec_id"]],
            axis=0,
        )
        distance, _ = extractor.kneighbors(np.array([mean]), return_distance=True)
        all_cluster.loc[all_cluster["frame_cluster"] == cluster_id, "distance"] = (
            distance[0][0]
        )
    # frame_clusterごとにsourceがc4の割合とサイズを計算

    group = all_cluster.groupby("frame_cluster")
    c4_ratio = group["source"].apply(lambda x: (x == "c4").mean())
    cluster_size = group.size()
    framenet_size = group["source"].apply(lambda x: (x == "framenet").sum())
    c4_size = group["source"].apply(lambda x: (x == "c4").sum())
    lu_counts = group["lu_name"].value_counts()
    lu_counts = lu_counts.groupby(level=0).apply(
        lambda x: [{key[1]: x[key]} for key in x.keys()]
    )
    distance = group["distance"].mean()

    # データフレームにまとめる
    distance_df = pd.DataFrame(
        {
            "distance": distance,
            "c4_ratio": c4_ratio,
            "size": cluster_size,
            "framenet_size": framenet_size,
            "c4_size": c4_size,
            "lu_counts": lu_counts,
        }
    )
    # c4_ratio_df = c4_ratio_df[c4_ratio_df["c4_ratio"] >= 0.95]
    # by = ["c4_size"]
    by = ["distance", "c4_size"]
    # c4_ratioで降順ソート、同率の場合はsizeで降順ソート
    distance_df = distance_df[distance_df["c4_size"] > 10]
    distance_sorted = distance_df.sort_values(by=by, ascending=[False] * len(by))

    output_txt += f"{distance_sorted[:top_n].to_markdown()}\n\n"

    for i, cluster_id in enumerate(distance_sorted.index[:top_n]):
        output_txt += f"{i} verb_form: {verb_form} add_method: {add_method} c4_rate: {c4_rate} setting: {setting} model_name: {model_name} clustering_method: {clustering_method} i: {i} cluster_id: {cluster_id}\n"

        cluster = all_cluster[all_cluster["frame_cluster"] == cluster_id].copy()
        cluster["text_widx"] = cluster.apply(
            lambda row: " ".join(
                np.insert(
                    row["text_widx"].split(),
                    [row["target_widx"], row["target_widx"] + 1],
                    ["**", "**"],
                )
            ),
            axis=1,
        )
        cluster = cluster.sort_values(by=["source", "lu_name"], ascending=False)
        cluster = cluster[["frame", "lu_name", "text_widx"]]
        output_txt += f"{cluster}\n\n"

    return output_txt


def main():
    verb_forms = ["lemma", "original"]
    add_methods = ["ratio", "frequency_100"]
    add_keys = ["verb", "lu_name"]
    clustering_datasets = ["c4first", "mix"]
    c4_rates = [0, 1, 2]
    settings = ["all_3_0", "all_3_1", "all_3_2"]
    # settings = ["all_3_1"]
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
        verb_forms[1],
        add_methods[0],
        add_keys[0],
        clustering_datasets[0],
        c4_rates[1],
        settings,
        model_names[2],
        clustering_methods[0],
        vec_type="wm",
        top_n=20,
    )

    # txt = get_distinct_clusters(
    #     verb_forms[1],
    #     add_methods[3],
    #     c4_rates[1],
    #     settings[1],
    #     model_names[2],
    #     clustering_methods[0],
    #     top_n=20,
    # )

    with open("./viewdata/cluster.txt", "w") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
