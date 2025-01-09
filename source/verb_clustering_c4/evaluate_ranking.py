import argparse
from pathlib import Path

from sfidml.f_induc.embedding import read_embedding
from sfidml.f_induc.ranking import SimilarityRanking
from sfidml.modules.score_ranking import (
    aggregate_ranking_scores,
    calculate_ranking_scores,
)
from sfidml.utils.data_utils import read_json, write_json, write_jsonl
from sfidml.utils.model_utils import fix_seed


def main(args):
    fix_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    best_params = read_json(args.input_params_file)
    df_vec, vec_array = read_embedding(
        args.input_dir,
        args.split,
        best_params["vec_type2run_number"],
        best_params["alpha"],
    )
    sr = SimilarityRanking(args.ranking_method)
    rankings = sr.ranking(df_vec, vec_array)
    scores = calculate_ranking_scores(rankings)
    metrics = aggregate_ranking_scores(scores)
    write_jsonl(rankings, args.output_dir / f"ranking_{args.split}.jsonl")
    write_jsonl(scores, args.output_dir / f"score_{args.split}.jsonl")
    write_json(metrics, args.output_dir / f"metrics_{args.split}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--input_params_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--ranking_method", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    args = parser.parse_args()
    print(args)
    main(args)
