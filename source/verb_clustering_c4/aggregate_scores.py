import statistics
from argparse import ArgumentParser, Namespace
from pathlib import Path

from sfidml.utils.data_utils import read_json, write_json


def main(args: Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}
    for i, input_dir in enumerate(args.input_dirs):
        metrics_i = read_json(input_dir / f"metrics_{args.split}.json")
        if (input_dir / "params.json").exists():
            params = read_json(input_dir / "params.json")
            if "alpha" in params:
                metrics_i["alpha"] = params["alpha"]
        for k, v in metrics_i.items():
            if i == 0:
                metrics[k] = [v]
            else:
                metrics[k].append(v)

    aggregated_metrics = {}
    for k, v in metrics.items():
        mean_v = statistics.mean(v)
        pstdev_v = statistics.pstdev(v)
        n_digits = 1 if mean_v >= 1 else 4
        aggregated_metrics["ave-" + k] = round(mean_v, n_digits)
        aggregated_metrics["sd-" + k] = round(pstdev_v, n_digits)

    write_json(aggregated_metrics, args.output_dir / f"metrics_{args.split}.json")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_dirs", type=Path, nargs="*", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True)
    args = parser.parse_args()
    print(args)
    main(args)
