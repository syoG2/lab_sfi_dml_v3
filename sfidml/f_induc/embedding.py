import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sfidml.f_induc.collate_fn import base_collate_fn
from sfidml.f_induc.dataset import BaseDataset
from sfidml.utils.data_utils import read_jsonl


class BaseEmbedding:
    def __init__(self, model, pretrained_model_name, vec_type, batch_size):
        self.model = model
        self.pretrained_model_name = pretrained_model_name
        self.vec_type = vec_type
        self.batch_size = batch_size

    def get_embedding(self, df):
        ds = BaseDataset(df, self.pretrained_model_name, self.vec_type)
        dl = DataLoader(
            ds,
            batch_size=self.batch_size,
            collate_fn=base_collate_fn,
            shuffle=False,
        )

        vec_list = []
        for batch in tqdm(dl):
            with torch.no_grad():
                vec_list += list(self.model(batch).cpu().detach().numpy())

        df_vec = (
            df.reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "vec_id"})
        )
        vec_array = np.array(vec_list)
        return df_vec, vec_array


def read_embedding(vec_dir, split, vec_type2run_number, alpha):
    if alpha == 0:
        dir1 = vec_dir / "word" / vec_type2run_number["word"]
        vec_array = np.load(dir1 / f"vec_{split}.npz", allow_pickle=True)["vec"]
        ex_file = dir1 / f"exemplars_{split}.jsonl"
    elif alpha == 1:
        dir1 = vec_dir / "mask" / vec_type2run_number["mask"]
        vec_array = np.load(dir1 / f"vec_{split}.npz", allow_pickle=True)["vec"]
        ex_file = dir1 / f"exemplars_{split}.jsonl"
    else:
        dir1 = vec_dir / "word" / vec_type2run_number["word"]
        dir2 = vec_dir / "mask" / vec_type2run_number["mask"]
        word_array = np.load(dir1 / f"vec_{split}.npz", allow_pickle=True)[
            "vec"
        ]
        mask_array = np.load(dir2 / f"vec_{split}.npz", allow_pickle=True)[
            "vec"
        ]
        vec_array = word_array * (1 - alpha) + mask_array * alpha
        ex_file = dir1 / f"exemplars_{split}.jsonl"
    df_vec = pd.DataFrame(read_jsonl(ex_file))
    return df_vec, vec_array
