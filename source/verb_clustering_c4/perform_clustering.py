import argparse
from pathlib import Path

from sfidml.f_induc.clustering_onestep import OnestepClustering
from sfidml.f_induc.clustering_twostep import TwostepClustering
from sfidml.f_induc.clustering_twostep_lu import TwostepClusteringLU
from sfidml.f_induc.embedding import read_embedding
from sfidml.utils.data_utils import read_json, write_json, write_jsonl
from sfidml.utils.model_utils import fix_seed


def main(args):
    fix_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_params_file is not None:
        best_params = read_json(args.input_params_file)
        for key, value in best_params.items():
            setattr(args, key, value)
    else:
        if args.alpha == 0:
            args.vec_type2run_number = {"word": args.run_number}
        elif args.alpha == 1:
            args.vec_type2run_number = {"mask": args.run_number}
        else:
            run_number_word, run_number_mask = args.run_number.split("-")
            args.vec_type2run_number = {
                "word": run_number_word,
                "mask": run_number_mask,
            }

    if args.clustering_name == "onestep":
        clustering = OnestepClustering(args.clustering_method)
    elif args.clustering_name == "twostep":
        clustering = TwostepClustering(args.clustering_method1, args.clustering_method2)
    elif args.clustering_name == "twostep_lu":
        clustering = TwostepClusteringLU(
            args.clustering_method1, args.clustering_method2
        )

    df_vec, vec_array = read_embedding(
        args.input_dir, "dev", args.vec_type2run_number, args.alpha
    )
    params = clustering.make_params(df_vec, vec_array)

    if args.clustering_name == "onestep":
        df_clu_dev = clustering.step(df_vec, vec_array, params)
    elif args.clustering_name == "twostep":
        df_clu_dev = clustering.step(df_vec, vec_array, vec_array, params)
    elif args.clustering_name == "twostep_lu":
        df_clu_dev = clustering.step(df_vec, vec_array, vec_array, params)

    # [ ]:C4のみで先にクラスタリングする場合、splitの部分を変更する
    if args.c4first:
        df_vec, vec_array = read_embedding(
            args.input_dir, "test-c4", args.vec_type2run_number, args.alpha
        )
    else:
        df_vec, vec_array = read_embedding(
            args.input_dir, "test", args.vec_type2run_number, args.alpha
        )

    if args.clustering_name == "onestep":
        df_clu_test = clustering.step(df_vec, vec_array, params)
    elif args.clustering_name == "twostep":
        df_clu_test = clustering.step(df_vec, vec_array, vec_array, params)
    elif args.clustering_name == "twostep_lu":
        df_clu_test = clustering.step(df_vec, vec_array, vec_array, params)

    write_jsonl(
        df_clu_dev.to_dict("records"),
        (args.output_dir / "exemplars_dev.jsonl"),
    )

    # [ ]:C4のみで先にクラスタリングする場合、出力ファイルをexemplars_test-c4.jsonlに変更する
    if args.c4first:
        write_jsonl(
            df_clu_test.to_dict("records"),
            (args.output_dir / "exemplars_test-c4.jsonl"),
        )
    else:
        write_jsonl(
            df_clu_test.to_dict("records"),
            (args.output_dir / "exemplars_test.jsonl"),
        )
    write_json(vars(args), args.output_dir / "params.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument(
        "--clustering_name",
        type=str,
        choices=["onestep", "twostep", "twostep_lu"],
        required=True,
    )

    parser.add_argument("--input_params_file", type=Path, required=False)

    parser.add_argument("--alpha", type=float, required=False)
    parser.add_argument("--run_number", type=str, required=False)

    parser.add_argument(
        "--clustering_method",
        type=str,
        choices=["average", "ward"],
        required=False,
    )
    parser.add_argument(
        "--clustering_method1",
        type=str,
        choices=["xmeans", "average", "axmeans"],
        required=False,
    )
    parser.add_argument(
        "--clustering_method2",
        type=str,
        choices=["average", "ward"],
        required=False,
    )
    parser.add_argument("--c4first", type=bool, default=False)
    args = parser.parse_args()
    print(args)
    main(args)
