import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sfidml.utils.data_utils import read_jsonl, write_jsonl


def _make_n_splits(v_list, n_splits):
    idx_list = np.array_split(range(len(v_list)), n_splits)
    return [[v_list[i] for i in idx] for idx in idx_list]


def _make_verb_list(df, n_splits):
    all_list, v2_list = [], []
    for verb, _ in df.groupby(["verb", "verb_frame"]).count().index:
        if verb not in all_list:
            all_list.append(verb)
        else:
            if verb not in v2_list:
                v2_list.append(verb)
    v1_list = sorted(set(all_list) - set(v2_list))
    v2_list = sorted(v2_list)

    random.seed(0)
    random.shuffle(v1_list)
    random.shuffle(v2_list)
    n_v1_list = _make_n_splits(v1_list, n_splits)
    n_v2_list = _make_n_splits(v2_list, n_splits)[::-1]
    n_v_list = [v1 + v2 for v1, v2 in zip(n_v1_list, n_v2_list, strict=False)]
    return n_v_list


def decide_sets(df, setting_prefix, n_splits):
    df = df.copy()
    n_v_list = _make_verb_list(df, n_splits) * 2

    for i in tqdm(range(n_splits)):
        test_v_list = n_v_list[i]
        dev_v_list = n_v_list[i + 1]
        train_v_list = sum(n_v_list[i + 2 : i + n_splits], [])

        v_sets_dict = {v: "test" for v in test_v_list}
        v_sets_dict.update({v: "dev" for v in dev_v_list})
        v_sets_dict.update({v: "train" for v in train_v_list})

        setting = "_".join([setting_prefix, str(n_splits), str(i)])
        df[setting] = df["verb"].map(v_sets_dict)
        df[setting] = df[setting].fillna("disuse")
    return df


def read_c4_datasets(
    file_id_list: list[int],
    part_id_list: list[int],
    split_name_list: list[str],
    lu_name_list: np.ndarray,
    text_input_style: str = "token0",
) -> pd.DataFrame:
    ret_df = pd.DataFrame()
    for file_id in tqdm(file_id_list):
        for part_id in tqdm(part_id_list, leave=False):
            for split_name in tqdm(split_name_list, leave=False):
                input_file: Path = Path(
                    f"./data/preprocessing/c4/preprocess/{text_input_style}/{split_name}_{file_id:05}/lu/exemplar_{part_id}.jsonl"
                )
                if input_file.exists():
                    # df = pd.read_json(input_file, lines=True, engine="pyarrow")
                    df = pd.read_json(input_file, lines=True)
                    df = df[df["lu_name"].str.contains(r"^[a-zA-Z\s\W]+$", regex=True)]
                    df["target_widxs"] = df["target_widx"]
                    df["target_widx"] = df["target_widx_head"].apply(lambda x: x[2])
                    df = df[
                        [
                            "ex_idx",
                            "verb",
                            "frame",
                            "verb_frame",
                            "text_widx",
                            "target_widx",
                            "target_widxs",
                            "preprocessed_lu_idx",
                            "source",
                            "lu_name",
                            "target_widx_head",
                        ]
                    ]

                    if len(df) > 0:
                        df = df[df["lu_name"].isin(lu_name_list)]
                        ret_df = pd.concat([ret_df, df], ignore_index=True)
                else:
                    tqdm.write(f"{input_file} is not found.")

    return ret_df


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(read_jsonl(args.input_file))
    df["target_widxs"] = df["target_widx"]
    df["target_widx"] = df["target_widx_head"].apply(lambda x: x[2])
    df["frame"] = df["frame_name"]
    df["verb_frame"] = df["verb"].str.cat(df["frame"], sep=":")
    df = df[
        [
            "ex_idx",
            "verb",
            "frame",
            "verb_frame",
            "text_widx",
            "target_widx",
            "target_widxs",
            "preprocessed_lu_idx",
            "source",
            "lu_name",
            "target_widx_head",
        ]
    ]
    # df.loc[:, ["source"]] = "framenet"

    df = decide_sets(df, args.setting_prefix, args.n_splits)

    if args.c4_rate > 0:
        settings = [
            f"{args.setting_prefix}_{args.n_splits}_{n}" for n in range(args.n_splits)
        ]
        lu_name_list = [
            list(df[df[setting] == "test"]["lu_name"].unique()) for setting in settings
        ]
        verb_list = [
            list(df[df[setting] == "test"]["verb"].unique()) for setting in settings
        ]
        file_id_list = [0, 1, 2, 3, 4, 5]
        part_id_list = list(range(350))
        split_name_list = ["train"]
        text_input_style = "token0"

        for file_id in tqdm(file_id_list):
            for part_id in tqdm(part_id_list, leave=False):
                for split_name in tqdm(split_name_list, leave=False):
                    input_file: Path = Path(
                        f"./data/preprocessing/c4/preprocess/{text_input_style}/{split_name}_{file_id:05}/lu/exemplar_{part_id}.jsonl"
                    )
                    if input_file.exists():
                        # df_c4 = pd.read_json(input_file, lines=True, engine="pyarrow")
                        df_c4 = pd.read_json(input_file, lines=True)
                        df_c4["lu_name"] = df_c4["lu_name"].str.replace(
                            r"\s*-\s*", "-", regex=True
                        )
                        df_c4 = df_c4[df_c4["lu_name"].isin(df["lu_name"].unique())]
                        df_c4["target_widxs"] = df_c4["target_widx"]
                        df_c4["target_widx"] = df_c4["target_widx_head"].apply(
                            lambda x: x[2]
                        )
                        df_c4 = df_c4[
                            [
                                "ex_idx",
                                "verb",
                                "frame",
                                "verb_frame",
                                "text_widx",
                                "target_widx",
                                "target_widxs",
                                "preprocessed_lu_idx",
                                "source",
                                "lu_name",
                                "target_widx_head",
                            ]
                        ]
                        df_c4[settings] = "disuse"

                        if (args.add_method == "ratio") | (
                            args.add_method == "c4first"
                        ):
                            for n in tqdm(range(args.n_splits), leave=False):
                                setting = settings[n]
                                additional = df_c4[
                                    df_c4["lu_name"].isin(lu_name_list[n])
                                ]
                                additional.loc[:, setting] = "test"
                                df = pd.concat([df, additional], ignore_index=True)

                                # 必要なデータのみをフィルタリング
                                test_df = df[
                                    (df[setting] == "test")
                                    & df["lu_name"].isin(lu_name_list[n])
                                ]

                                # グループ化してカウントを取得
                                counts = (
                                    test_df.groupby(["lu_name", "source"])
                                    .size()
                                    .unstack(fill_value=0)
                                )

                                # framenet_count と c4_count を取得
                                counts["framenet_count"] = counts.get("framenet", 0)
                                counts["c4_count"] = counts.get("c4", 0)

                                # フラグを作成して削除対象を特定
                                # c4のデータが十分集まったLUはこれ以降追加しない
                                lu_to_remove = counts[
                                    counts["framenet_count"] * args.c4_rate
                                    <= counts["c4_count"]
                                ].index.tolist()

                                # 一括で削除
                                lu_name_list[n] = [
                                    lu
                                    for lu in lu_name_list[n]
                                    if lu not in lu_to_remove
                                ]
                        elif (args.add_method == "ratio_verb") | (
                            args.add_method == "c4first_verb"
                        ):
                            for n in tqdm(range(args.n_splits), leave=False):
                                setting = settings[n]
                                additional = df_c4[df_c4["verb"].isin(verb_list[n])]
                                additional.loc[:, setting] = "test"
                                df = pd.concat([df, additional], ignore_index=True)

                                # 必要なデータのみをフィルタリング
                                test_df = df[
                                    (df[setting] == "test")
                                    & df["verb"].isin(verb_list[n])
                                ]

                                # グループ化してカウントを取得
                                counts = (
                                    test_df.groupby(["verb", "source"])
                                    .size()
                                    .unstack(fill_value=0)
                                )

                                # framenet_count と c4_count を取得
                                counts["framenet_count"] = counts.get("framenet", 0)
                                counts["c4_count"] = counts.get("c4", 0)

                                # フラグを作成して削除対象を特定
                                # c4のデータが十分集まったLUはこれ以降追加しない
                                verb_to_remove = counts[
                                    counts["framenet_count"] * args.c4_rate
                                    <= counts["c4_count"]
                                ].index.tolist()

                                # 一括で削除
                                verb_list[n] = [
                                    lu
                                    for lu in verb_list[n]
                                    if lu not in verb_to_remove
                                ]

                        elif args.add_method == "sequential_n_verb":
                            for n in tqdm(range(args.n_splits), leave=False):
                                setting = settings[n]
                                additional = df_c4[df_c4["verb"].isin(verb_list[n])]
                                additional.loc[:, setting] = "test"
                                df = pd.concat([df, additional], ignore_index=True)

                                # 必要なデータのみをフィルタリング
                                test_df = df[
                                    (df[setting] == "test")
                                    & df["verb"].isin(verb_list[n])
                                ]

                                # グループ化してカウントを取得
                                counts = (
                                    test_df.groupby(["verb", "source"])
                                    .size()
                                    .unstack(fill_value=0)
                                )

                                # framenet_count と c4_count を取得
                                counts["framenet_count"] = counts.get("framenet", 0)
                                counts["c4_count"] = counts.get("c4", 0)

                                # フラグを作成して削除対象を特定
                                # c4のデータが十分集まったLUはこれ以降追加しない
                                verb_to_remove = counts[
                                    args.maximum_verb <= counts["c4_count"]
                                ].index.tolist()

                                # 一括で削除
                                verb_list[n] = [
                                    lu
                                    for lu in verb_list[n]
                                    if lu not in verb_to_remove
                                ]
                        elif args.add_method == "sequential":
                            ok = True
                            for n in tqdm(range(args.n_splits), leave=False):
                                setting = settings[n]
                                framenet_count = len(
                                    df[
                                        (df[setting] == "test")
                                        & (df["source"] == "framenet")
                                    ]
                                )
                                c4_count = len(
                                    df[(df[setting] == "test") & (df["source"] == "c4")]
                                )
                                # tqdm.write(f"{n}, {framenet_count}, {c4_count}")
                                if framenet_count * args.c4_rate > c4_count:
                                    additional = df_c4[
                                        df_c4["lu_name"].isin(lu_name_list[n])
                                    ]
                                    additional.loc[:, setting] = "test"
                                    df = pd.concat([df, additional], ignore_index=True)
                                    ok = False
                            if ok:
                                break
                        elif args.add_method == "sequential_verb":
                            ok = True
                            for n in tqdm(range(args.n_splits), leave=False):
                                setting = settings[n]
                                framenet_count = len(
                                    df[
                                        (df[setting] == "test")
                                        & (df["source"] == "framenet")
                                    ]
                                )
                                c4_count = len(
                                    df[(df[setting] == "test") & (df["source"] == "c4")]
                                )
                                # tqdm.write(f"{n}, {framenet_count}, {c4_count}")
                                if framenet_count * args.c4_rate > c4_count:
                                    additional = df_c4[df_c4["verb"].isin(verb_list[n])]
                                    additional.loc[:, setting] = "test"
                                    df = pd.concat([df, additional], ignore_index=True)
                                    ok = False
                            if ok:
                                break
                    else:
                        tqdm.write(f"{input_file} is not found.")
                else:
                    continue
                break
            else:
                continue
            break

        if (args.add_method == "ratio") | (args.add_method == "c4first"):
            incomplete_lus: list[list[str]] = [[] for _ in range(args.n_splits)]
            for n in tqdm(range(args.n_splits)):
                setting = settings[n]
                for lu_name in tqdm(df[df[setting] == "test"]["lu_name"].unique()):
                    framenet_count = df[
                        (df["source"] == "framenet")
                        & (df[setting] == "test")
                        & (df["lu_name"] == lu_name)
                    ].shape[0]
                    c4_count = df[
                        (df["source"] == "c4")
                        & (df[setting] == "test")
                        & (df["lu_name"] == lu_name)
                    ].shape[0]
                    if c4_count < framenet_count * args.c4_rate:
                        tqdm.write(
                            f"Warning: {lu_name}: {args.c4_rate} * {framenet_count} > {c4_count}"
                        )
                        incomplete_lus[n].append(
                            f"{lu_name}: {args.c4_rate} * {framenet_count} > {c4_count}"
                        )
                        if args.drop:
                            df[((df[setting] == "test") & (df["lu_name"] == lu_name))][
                                setting
                            ] = "disuse"
                    else:
                        remove = df[
                            (df["source"] == "c4")
                            & (df[setting] == "test")
                            & (df["lu_name"] == lu_name)
                        ].tail(c4_count - (framenet_count * args.c4_rate))
                        df = df.drop(remove.index)
            for n in tqdm(range(args.n_splits)):
                setting = f"{args.setting_prefix}_{args.n_splits}_{n}"
                output_dir = args.output_dir / str(args.c4_rate) / setting
                output_dir.mkdir(parents=True, exist_ok=True)

                with open(output_dir / "incomplete_lus.txt", "w") as f:
                    for lu in incomplete_lus[n]:
                        print(lu, file=f)
        elif (args.add_method == "ratio_verb") | (args.add_method == "c4first_verb"):
            incomplete_verbs: list[list[str]] = [[] for _ in range(args.n_splits)]
            for n in tqdm(range(args.n_splits)):
                setting = settings[n]
                for verb in tqdm(df[df[setting] == "test"]["verb"].unique()):
                    framenet_count = df[
                        (df["source"] == "framenet")
                        & (df[setting] == "test")
                        & (df["verb"] == verb)
                    ].shape[0]
                    c4_count = df[
                        (df["source"] == "c4")
                        & (df[setting] == "test")
                        & (df["verb"] == verb)
                    ].shape[0]
                    if c4_count < framenet_count * args.c4_rate:
                        tqdm.write(
                            f"Warning: {verb}: {args.c4_rate} * {framenet_count} > {c4_count}"
                        )
                        incomplete_verbs[n].append(
                            f"{verb}: {args.c4_rate} * {framenet_count} > {c4_count}"
                        )
                        if args.drop:
                            df[((df[setting] == "test") & (df["verb"] == verb))][
                                setting
                            ] = "disuse"
                    else:
                        remove = df[
                            (df["source"] == "c4")
                            & (df[setting] == "test")
                            & (df["verb"] == verb)
                        ].tail(c4_count - (framenet_count * args.c4_rate))
                        df = df.drop(remove.index)
            for n in tqdm(range(args.n_splits)):
                setting = f"{args.setting_prefix}_{args.n_splits}_{n}"
                output_dir = args.output_dir / str(args.c4_rate) / setting
                output_dir.mkdir(parents=True, exist_ok=True)

                with open(output_dir / "incomplete_lus.txt", "w") as f:
                    for verb in incomplete_verbs[n]:
                        print(verb, file=f)
        elif args.add_method == "sequential_n_verb":
            for n in tqdm(range(args.n_splits)):
                setting = settings[n]
                for verb in tqdm(df[df[setting] == "test"]["verb"].unique()):
                    framenet_count = df[
                        (df["source"] == "framenet")
                        & (df[setting] == "test")
                        & (df["verb"] == verb)
                    ].shape[0]
                    c4_count = df[
                        (df["source"] == "c4")
                        & (df[setting] == "test")
                        & (df["verb"] == verb)
                    ].shape[0]
                    if c4_count > args.maximum_verb:
                        remove = df[
                            (df["source"] == "c4")
                            & (df[setting] == "test")
                            & (df["verb"] == verb)
                        ].tail(c4_count - (args.maximum_verb))
                        df = df.drop(remove.index)
            for n in tqdm(range(args.n_splits)):
                setting = f"{args.setting_prefix}_{args.n_splits}_{n}"
                output_dir = args.output_dir / str(args.c4_rate) / setting
                output_dir.mkdir(parents=True, exist_ok=True)

        elif args.add_method == "sequential":
            for n in tqdm(range(args.n_splits)):
                setting = settings[n]
                framenet_count = len(
                    df[(df[setting] == "test") & (df["source"] == "framenet")]
                )
                c4_count = len(df[(df[setting] == "test") & (df["source"] == "c4")])
                if framenet_count * args.c4_rate > c4_count:
                    tqdm.write(
                        f"Warning: {setting}: {args.c4_rate} * {framenet_count} > {c4_count}"
                    )
                else:
                    remove = df[(df["source"] == "c4") & (df[setting] == "test")].tail(
                        c4_count - framenet_count * args.c4_rate
                    )
                    df = df.drop(remove.index)
        elif args.add_method == "sequential_verb":
            for n in tqdm(range(args.n_splits)):
                setting = settings[n]
                framenet_count = len(
                    df[(df[setting] == "test") & (df["source"] == "framenet")]
                )
                c4_count = len(df[(df[setting] == "test") & (df["source"] == "c4")])
                if framenet_count * args.c4_rate > c4_count:
                    tqdm.write(
                        f"Warning: {setting}: {args.c4_rate} * {framenet_count} > {c4_count}"
                    )
                else:
                    remove = df[(df["source"] == "c4") & (df[setting] == "test")].tail(
                        c4_count - (framenet_count * args.c4_rate)
                    )
                    df = df.drop(remove.index)

    for n in tqdm(range(args.n_splits)):
        setting = f"{args.setting_prefix}_{args.n_splits}_{n}"
        output_dir = args.output_dir / str(args.c4_rate) / setting
        output_dir.mkdir(parents=True, exist_ok=True)

        if (args.add_method == "c4first") | (args.add_method == "c4first_verb"):
            for split in ["test-framenet", "test-c4", "dev", "train"]:
                if split == "test-framenet":
                    df_split = df[
                        (df[setting] == "test") & (df["source"] == "framenet")
                    ]
                elif split == "test-c4":
                    df_split = df[(df[setting] == "test") & (df["source"] == "c4")]
                else:
                    df_split = df[df[setting] == split]
                write_jsonl(
                    df_split.to_dict("records"),
                    output_dir / f"exemplars_{split}.jsonl",
                )
        else:
            for split in ["test", "dev", "train"]:
                df_split = df[df[setting] == split]
                write_jsonl(
                    df_split.to_dict("records"),
                    output_dir / f"exemplars_{split}.jsonl",
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--setting_prefix", type=str, default="all")
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--c4_rate", type=int, default=0)
    parser.add_argument(
        "--add_method",
        type=str,
        choices=[
            "sequential",
            "ratio",
            "c4first",
            "ratio_verb",
            "sequential_verb",
            "c4first_verb",
            "sequential_n_verb",
        ],
    )
    parser.add_argument("--maximum_verb", type=int, default=100)
    parser.add_argument("--drop", type=bool, default=False)
    args = parser.parse_args()
    print(args)
    main(args)
