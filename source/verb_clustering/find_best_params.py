import argparse
from pathlib import Path

from tqdm import tqdm

from sfidml.f_induc.clustering_onestep import OnestepClustering
from sfidml.f_induc.clustering_twostep import TwostepClustering
from sfidml.f_induc.embedding import read_embedding
from sfidml.modules.score_clustering import calculate_bcubed
from sfidml.utils.data_utils import write_json
from sfidml.utils.model_utils import fix_seed


def main(args):
    fix_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.vec_type == "word":
        alpha_list = [0]
    elif args.vec_type == "mask":
        alpha_list = [1]
    elif args.vec_type == "wm":
        alpha_list = [0, 1]
        alpha_list2 = [i / 10 for i in range(11)]

    if args.clustering_name == "onestep":
        clustering = OnestepClustering(args.clustering_method)
    elif args.clustering_name == "twostep":
        clustering = TwostepClustering(
            args.clustering_method1, args.clustering_method2
        )

    best_runs = {}
    for alpha in tqdm(alpha_list):
        best_bcf = 0
        for run_number in tqdm(args.run_numbers):
            runs = {alpha: run_number}
            df_vec, vec_array = read_embedding(
                args.input_dir, "dev", alpha, runs
            )
            clustering.make_params(df_vec, vec_array)

            if args.clustering_name == "onestep":
                df_output = clustering.step(df_vec, vec_array)
            elif args.clustering_name == "twostep":
                df_output = clustering.step(df_vec, vec_array, vec_array)

            true = df_output.groupby("frame")["ex_idx"].agg(list).tolist()
            pred = (
                df_output.groupby("frame_cluster")["ex_idx"].agg(list).tolist()
            )
            bcf = calculate_bcubed(true, pred)[2]
            if best_bcf < bcf:
                best_bcf = bcf
                best_runs[alpha] = run_number

    if args.vec_type == "wm":
        best_bcf = 0
        for alpha in tqdm(alpha_list2):
            df_vec, vec_array = read_embedding(
                args.input_dir, "dev", alpha, best_runs
            )
            clustering.make_params(df_vec, vec_array)

            if args.clustering_name == "onestep":
                df_output = clustering.step(df_vec, vec_array)
            elif args.clustering_name == "twostep":
                df_output = clustering.step(df_vec, vec_array, vec_array)

            true = df_output.groupby("frame")["ex_idx"].agg(list).tolist()
            pred = (
                df_output.groupby("frame_cluster")["ex_idx"].agg(list).tolist()
            )
            bcf = calculate_bcubed(true, pred)[2]
            if best_bcf < bcf:
                best_bcf = bcf
                best_alpha = alpha
    else:
        best_alpha = alpha

    best_params = {"alpha": best_alpha, "runs": best_runs}
    if args.clustering_name == "onestep":
        best_params["clustering_method"] = args.clustering_method
    elif args.clustering_name == "twostep":
        best_params["clustering_method1"] = args.clustering_method1
        best_params["clustering_method2"] = args.clustering_method2

    write_json(best_params, args.output_dir / "best_params.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--vec_type", type=str, choices=["word", "mask", "wm"])
    parser.add_argument("--run_numbers", type=str, nargs="*", default=["00"])

    parser.add_argument(
        "--clustering_name", type=str, choices=["onestep", "twostep"]
    )

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
    args = parser.parse_args()
    print(args)
    main(args)
