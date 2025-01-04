import argparse
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from datasets import load_dataset
from pydantic import BaseModel, Field
from tqdm import tqdm


class C4Id(BaseModel):
    # c4の元データの参照に必要な情報を格納
    split_name: str
    file_id: int
    timestamp: datetime
    url: str
    ex_idx: UUID = Field(default_factory=lambda: uuid4())  # C4のデータがtimestampとurlで一意に決まるかが不明なためuuidを追加


class RawC4Data(BaseModel):
    # c4の元データを格納
    id_data: C4Id
    text: str
    source: str = "c4"


def main(args):
    # outputディレクトリの作成
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "json",
        data_files=(
            "https://huggingface.co/datasets/allenai/c4/resolve/main/en/"
            f"c4-{args.split_name}.{args.file_id:05}-of-01024.json.gz"
        ),
    )

    text_list: list[str] = dataset[args.split_name]["text"]
    timestamp_list: list[datetime] = dataset[args.split_name]["timestamp"]
    url_list: list[str] = dataset[args.split_name]["url"]

    exemplars: list[RawC4Data] = []
    for text, timestamp, url in zip(text_list, timestamp_list, url_list, strict=False):
        id_data: C4Id = C4Id(
            split_name=args.split_name,
            file_id=args.file_id,
            timestamp=timestamp,
            url=url,
        )
        exemplar: RawC4Data = RawC4Data(
            id_data=id_data,
            text=text,
        )
        exemplars.append(exemplar)

    with open(args.output_file, "w") as f:
        with tqdm(exemplars) as pbar:
            pbar.set_description("[write exemplars]")
            for exemplar in pbar:
                print(exemplar.model_dump_json(), file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_id", type=int, required=True)
    parser.add_argument("--split_name", type=str, choices=["train", "validation"], required=False)
    parser.add_argument("--output_file", type=Path, required=True)
    args = parser.parse_args()
    print(args)
    main(args)


# class Args(BaseModel):
#     file_id: int
#     split_name: str = "train"  # "train" or "validation"
#     output_file: Path = Path("")

#     def model_post_init(self, __context):
#         if self.output_file == Path(""):
#             self.output_file = Path(f"./data/c4/{self.split_name}_{self.file_id:05}.jsonl")
