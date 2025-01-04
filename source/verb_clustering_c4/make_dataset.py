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
                    df = pd.read_json(input_file, lines=True)
                    df = df[
                        [
                            "ex_idx",
                            "verb",
                            "frame",
                            "verb_frame",
                            "text_widx",
                            "target_widx",
                            "preprocessed_lu_idx",
                            "source",
                            "lu_name",
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
            "preprocessed_lu_idx",
            "source",
            "lu_name",
        ]
    ]
    # df.loc[:, ["source"]] = "framenet"

    df = decide_sets(df, args.setting_prefix, args.n_splits)

    if args.c4_rate > 0:
        df_c4 = read_c4_datasets(
            [0, 1, 2, 3, 4],
            list(range(350)),
            ["train"],
            df["lu_name"].unique(),
        )
        for n_tmp in tqdm(range(args.n_splits)):
            setting_tmp = f"{args.setting_prefix}_{args.n_splits}_{n_tmp}"
            df_c4.loc[:, [setting_tmp]] = "unused"

        with tqdm(range(args.n_splits)) as pbar:
            for n in pbar:
                setting = f"{args.setting_prefix}_{args.n_splits}_{n}"
                df_c4_mutched = pd.DataFrame()

                with tqdm(
                    df[df[setting] == "test"]["lu_name"].value_counts().items()
                ) as pbar2:
                    for lu_name, count in pbar2:
                        # tqdm.write(f"{lu_name}:{count}")
                        df_c4_tmp = df_c4[df_c4["lu_name"] == lu_name]
                        df_c4_tmp.loc[:, [setting]] = "test"
                        if len(df_c4_tmp) < count:
                            tqdm.write(
                                f"Warning: {lu_name}: {len(df_c4_tmp)} < {count}"
                            )
                        else:
                            df_c4_tmp = df_c4_tmp.sample(count, random_state=0)
                        df_c4_mutched = pd.concat(
                            [df_c4_mutched, df_c4_tmp], ignore_index=True
                        )
                    df = pd.concat([df, df_c4_mutched], ignore_index=True)

    for n in tqdm(range(args.n_splits)):
        setting = f"{args.setting_prefix}_{args.n_splits}_{n}"
        output_dir = args.output_dir / setting
        output_dir.mkdir(parents=True, exist_ok=True)

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
    parser.add_argument("--c4_rate", type=int, default=1)
    args = parser.parse_args()
    print(args)
    main(args)
