import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sfidml.f_induc.collate_fn import base_collate_fn
from sfidml.f_induc.dataset import BaseDataset


def get_embedding(df, pretrained_model_name, vec_type, model, batch_size):
    dl = DataLoader(
        BaseDataset(df, pretrained_model_name, vec_type),
        batch_size=batch_size,
        collate_fn=base_collate_fn,
        shuffle=False,
    )

    vec_list = []
    for batch in tqdm(dl):
        with torch.no_grad():
            vec_list += list(model(batch).cpu().detach().numpy())

    df_vec = (
        df.reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "vec_id"})
    )
    vec_array = np.array(vec_list)
    return df_vec, vec_array
