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
    required_items_list: list[dict[str, int]],
    max_size_list: list[int],
    key: str,
    text_input_style: str = "token0",
) -> list[pd.DataFrame]:
    ret_df_list = [pd.DataFrame() for _ in range(len(required_items_list))]
    check_list = [False for _ in range(len(required_items_list))]
    with tqdm(file_id_list) as file_id_pbar:
        for file_id in file_id_pbar:
            for part_id in tqdm(part_id_list, leave=False):
                for split_name in tqdm(split_name_list, leave=False):
                    input_file: Path = Path(
                        f"./data/preprocessing/c4/preprocess/{text_input_style}/{split_name}_{file_id:05}/lu/exemplar_{part_id}.jsonl"
                    )
                    if input_file.exists():
                        df_c4 = pd.read_json(input_file, lines=True)
                        df_c4 = df_c4[
                            df_c4["lu_name"].str.contains(
                                r"^[a-zA-Z\s\W]+$", regex=True
                            )
                        ]  # remove non-english
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
                        for i, required_items in enumerate(required_items_list):
                            ret_df = ret_df_list[i]
                            if check_list[i]:
                                continue

                            filtered_df_c4 = df_c4[
                                df_c4[key].isin(required_items.keys())
                            ]
                            ret_df = pd.concat(
                                [ret_df, filtered_df_c4], ignore_index=True
                            )
                            ret_df = (
                                ret_df.groupby(key)[ret_df.columns]
                                .apply(
                                    lambda group: group.head(required_items[group.name])
                                )
                                .reset_index(drop=True)
                            )
                            if ret_df.shape[0] > max_size_list[i]:
                                remove_num = ret_df.shape[0] - max_size_list[i]
                                remove_ex_idxes = []
                                for _, row in tqdm(
                                    filtered_df_c4.iloc[::-1].iterrows()
                                ):
                                    if len(remove_ex_idxes) < remove_num:
                                        if row["ex_idx"] in ret_df["ex_idx"].values:
                                            remove_ex_idxes.append(row["ex_idx"])
                                    else:
                                        break
                                ret_df = ret_df[
                                    ~(ret_df["ex_idx"].isin(remove_ex_idxes))
                                ]
                                check_list[i] = True
                            ret_df_list[i] = ret_df
                        file_id_pbar.set_description(
                            f"{[ret_df.shape[0] for ret_df in ret_df_list]}"
                        )
                    else:
                        tqdm.write(f"{input_file} is not found.")
                    if all(
                        [
                            ret_df.groupby(key)[ret_df.columns]
                            .apply(
                                lambda group: group.shape[0]
                                == required_items[group.name]
                            )
                            .all()
                            for ret_df, required_items in zip(
                                ret_df_list, required_items_list
                            )
                        ]
                    ):
                        print("all clear")
                        return ret_df_list
                    if all(check_list):
                        print("all clear")
                        return ret_df_list

    return ret_df_list


def main(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.add_method != "ratio":
        frequency_max = int(args.add_method.split("_")[-1])

    df_framenet = pd.DataFrame(read_jsonl(args.input_file))
    df_framenet["target_widxs"] = df_framenet["target_widx"]
    df_framenet["target_widx"] = df_framenet["target_widx_head"].apply(lambda x: x[2])
    df_framenet["frame"] = df_framenet["frame_name"]
    df_framenet["verb_frame"] = df_framenet["verb"].str.cat(
        df_framenet["frame"], sep=":"
    )
    df_framenet = df_framenet[
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

    df = decide_sets(df_framenet, args.setting_prefix, args.n_splits)

    settings = [
        f"{args.setting_prefix}_{args.n_splits}_{n}" for n in range(args.n_splits)
    ]
    for setting in settings:
        print(df[df[setting] == "test"].shape[0])

    if args.c4_rate > 0:
        required_items_list = []
        for n in tqdm(range(args.n_splits)):
            setting = settings[n]
            if args.add_key == "verb":
                required_items = (
                    df[df[setting] == "test"].groupby("verb").size().to_dict()
                )
            elif args.add_key == "lu_name":
                required_items = (
                    df[df[setting] == "test"].groupby("lu_name").size().to_dict()
                )
            if args.add_method == "ratio":
                required_items = {
                    k: int(v * args.c4_rate) for k, v in required_items.items()
                }
            else:
                required_items = {k: frequency_max for k in required_items}
            required_items_list.append(required_items)

        df_c4_list = read_c4_datasets(
            [0, 1, 2, 3, 4, 5],
            list(range(350)),
            ["train"],
            required_items_list,
            max_size_list=[
                df[df[setting] == "test"].shape[0] * args.c4_rate
                for setting in settings
            ],
            key=args.add_key,
        )

        for n in tqdm(range(args.n_splits)):
            setting = settings[n]
            df_c4 = df_c4_list[n]
            output_dir = args.output_dir / str(args.c4_rate) / setting
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / f"incomplete_{args.add_key}.txt", "w") as f:
                for key, value in required_items_list[n].items():
                    c4_count = df_c4[df_c4[args.add_key] == key].shape[0]
                    if c4_count < value:
                        f.write(f"{key}: {value} > {c4_count}\n")

            df_c4_list[n][settings] = "disuse"
            df_c4_list[n][setting] = "test"
            df = pd.concat([df, df_c4_list[n]], ignore_index=True)

    for n in tqdm(range(args.n_splits)):
        setting = settings[n]
        output_dir = args.output_dir / str(args.c4_rate) / setting
        output_dir.mkdir(parents=True, exist_ok=True)

        for split in ["test-framenet", "test-c4", "test", "dev", "train"]:
            if split == "test-framenet":
                df_split = df[(df[setting] == "test") & (df["source"] == "framenet")]
            elif split == "test-c4":
                df_split = df[(df[setting] == "test") & (df["source"] == "c4")]
            elif split == "test":
                df_split = df[df[setting] == "test"]
            else:
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
    # parser.add_argument("--add_method", type=str, choices=["ratio", "frequency"])
    parser.add_argument("--add_method", type=str, default="frequency_100")
    parser.add_argument("--add_key", type=str, choices=["lu_name", "verb"])
    # parser.add_argument("--frequency_max", type=int, default=100)
    args = parser.parse_args()
    print(args)
    main(args)
