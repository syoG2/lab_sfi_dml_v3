import argparse
from pathlib import Path
from uuid import UUID, uuid4

from nltk.corpus import framenet
from pydantic import BaseModel, Field
from tqdm import tqdm


class FramenetId(BaseModel):
    id: int


class RawFramenetData(BaseModel):
    # 前処理前のframenetのデータ
    id_data: FramenetId  # 元データの参照に必要な情報を格納
    text: str  # 前処理をする前のtext
    target: list[list[int]]  # 前処理をする前のLUの位置(文字レベル)
    frame_name: str
    frame_id: int
    lu_name: str
    lu_id: int
    fe: list[list[list[int | str]] | dict[str, str]]
    source: str = "framenet"
    ex_idx: UUID = Field(default_factory=lambda: uuid4())


def make_exemplars(exemplars: any) -> list[RawFramenetData]:
    # 前処理前のframenetのデータを作成
    # nltk.corpus.framenet.exemplars()をRawFramenetDataのリストに変換
    ret: list[RawFramenetData] = []
    if hasattr(exemplars, "__iter__"):
        with tqdm() as pbar:
            pbar.set_description("[make_exemplars]")
            for exemplar in exemplars:
                try:
                    data: RawFramenetData = RawFramenetData(
                        id_data=FramenetId(id=exemplar.ID),
                        text=exemplar.text,
                        frame_name=exemplar.frame.name,
                        frame_id=exemplar.frame.ID,
                        lu_name=exemplar.LU.name,
                        lu_id=exemplar.LU.ID,
                        target=exemplar.Target,
                        fe=exemplar.FE,
                    )
                    ret.append(data)
                except KeyError as e:
                    print(f"key:'{e}'が存在しません。")
                pbar.update(1)
    else:
        print(f"fn.exemplars():'{type(exemplars)}'がlistではありません。")
    return ret


def main(args):
    # outputディレクトリの作成
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    exemplars: list[RawFramenetData] = make_exemplars(framenet.exemplars())
    with open(args.output_file, "w") as f:
        with tqdm(exemplars) as pbar:
            pbar.set_description("[write exemplars]")
            for exemplar in pbar:
                print(exemplar.model_dump_json(), file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=Path, default=Path("./data/preprocessing/framenet/collect/exemplars.jsonl"))
    args = parser.parse_args()
    print(args)
    main(args)
