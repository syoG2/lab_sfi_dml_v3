import argparse
from pathlib import Path

import pandas as pd

from sfidml.utils.model_utils import fix_seed


def main(args):
    fix_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_json(args.input_file, lines=True)
    df = df[
        [
            "featured_word",
            "text_widx",
            "preprocessed_lu_idx",
            "featured_word_idx",
            "lu_name",
        ]
    ]

    # 5分割交差検証用にデータを分割する
    # luの単語数が1のものと2以上のもので比率を合わせたい。
    # 学習データとテストデータに同じlu_nameが含まれないようにする
    df_lu_single = df[df["preprocessed_lu_idx"].apply(len) == 1]  # luの単語数が1のもの
    df_lu_multi = df[
        df["preprocessed_lu_idx"].apply(len) > 1
    ]  # luの単語数が2以上のもの

    if args.mode == "random":
        # LU抽出モデルを作成する際はランダムに分割して学習
        df_lu_single = df_lu_single.sample(frac=1, random_state=args.seed).reset_index(
            drop=True
        )
        df_lu_single_list: pd.DataFrame = [pd.DataFrame() for _ in range(args.n_splits)]
        for i in range(args.n_splits):
            df_lu_single_list[i] = df_lu_single.iloc[i :: args.n_splits].reset_index(
                drop=True
            )
        df_lu_single_list.sort(key=len)

        df_lu_multi = df_lu_multi.sample(frac=1, random_state=args.seed).reset_index(
            drop=True
        )
        df_lu_multi_list: pd.DataFrame = [pd.DataFrame() for _ in range(args.n_splits)]
        for i in range(args.n_splits):
            df_lu_multi_list[i] = df_lu_multi.iloc[i :: args.n_splits].reset_index(
                drop=True
            )
        df_lu_multi_list.sort(key=len, reverse=True)
    elif args.mode == "distinct":
        # "lu_name"でグループ化し、数が少ない順にソート
        lu_name_single_counts = df_lu_single["lu_name"].value_counts()
        df_lu_single_list: pd.DataFrame = [pd.DataFrame() for _ in range(args.n_splits)]
        # ソートされた順にfor文を回す
        for lu_name in lu_name_single_counts.index:
            df_lu_single_list.sort(key=len)
            df_lu_single_list[0] = pd.concat(
                [
                    df_lu_single_list[0],
                    df_lu_single[df_lu_single["lu_name"] == lu_name],
                ],
                ignore_index=True,
            )
        df_lu_single_list.sort(key=len)

        # "lu_name"でグループ化し、数が少ない順にソート
        lu_name_multi_counts = df_lu_multi["lu_name"].value_counts()
        df_lu_multi_list: pd.DataFrame = [pd.DataFrame() for _ in range(args.n_splits)]
        # ソートされた順にfor文を回す
        for lu_name in lu_name_multi_counts.index:
            df_lu_multi_list.sort(key=len)
            df_lu_multi_list[0] = pd.concat(
                [df_lu_multi_list[0], df_lu_multi[df_lu_multi["lu_name"] == lu_name]],
                ignore_index=True,
            )
        df_lu_multi_list.sort(key=len, reverse=True)

    df_list = [
        pd.concat([df_lu_single_list[i], df_lu_multi_list[i]], ignore_index=True)
        for i in range(args.n_splits)
    ]

    for i in range(args.n_splits):
        df_list[i].to_json(
            args.output_dir / f"{args.n_splits}_{i}.jsonl",
            lines=True,
            orient="records",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_splits", type=int, default=5)  # 分割数
    parser.add_argument(
        "--input_file",
        type=Path,
        default=Path("./data/preprocessing/framenet/preprocess/exemplars.jsonl"),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["distinct", "random"],
        required=False,
    )
    args = parser.parse_args()
    print(args)
    main(args)
