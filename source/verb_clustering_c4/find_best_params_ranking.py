import argparse
from pathlib import Path

from tqdm import tqdm

from sfidml.f_induc.embedding import read_embedding
from sfidml.f_induc.ranking import run_ranking
from sfidml.utils.data_utils import write_json
from sfidml.utils.model_utils import fix_seed


def main(args):
    fix_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.vec_type == "word":
        vec_types = ["word"]
        alphas = [0]
    elif args.vec_type == "mask":
        vec_types = ["mask"]
        alphas = [1]
    elif args.vec_type == "wm":
        vec_types = ["word", "mask"]
        alphas = [i / 10 for i in range(11)]

    best_vec_type2run_number = {}
    for vec_type in tqdm(vec_types):
        best_acc = 0
        for run_number in tqdm(args.run_numbers):
            vec_type2run_number = {vec_type: run_number}
            alpha = 0 if vec_type == "word" else 1
            df_vec, vec_array = read_embedding(
                args.input_dir, "dev", vec_type2run_number, alpha
            )
            acc = run_ranking(df_vec, vec_array, args.ranking_method)["acc"]
            if best_acc <= acc:
                best_acc = acc
                best_vec_type2run_number[vec_type] = run_number

    if args.vec_type == "wm":
        best_acc = 0
        for alpha in tqdm(alphas):
            df_vec, vec_array = read_embedding(
                args.input_dir, "dev", best_vec_type2run_number, alpha
            )
            acc = run_ranking(df_vec, vec_array, args.ranking_method)["acc"]
            if best_acc <= acc:
                best_acc = acc
                best_alpha = alpha
    else:
        best_alpha = alpha
    best_params = {
        "alpha": best_alpha,
        "vec_type2run_number": best_vec_type2run_number,
    }
    write_json(best_params, args.output_dir / "best_params.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument(
        "--vec_type", type=str, choices=["word", "mask", "wm"], required=True
    )
    parser.add_argument("--run_numbers", type=str, nargs="*", required=False)

    parser.add_argument(
        "--ranking_method", type=str, default="all-all", required=True
    )
    args = parser.parse_args()
    print(args)
    main(args)
