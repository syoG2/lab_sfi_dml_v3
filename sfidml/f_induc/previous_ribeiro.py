import networkx as nx
import numpy as np
import pandas as pd
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from chinese_whispers import aggregate_clusters, chinese_whispers
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset


def ribeiro_collate_fn(batch):
    output_dict = {
        "verb": [],
        "frame": [],
        "ex_idx": [],
        "batch_size": len(batch),
    }
    output_dict["input_ids"] = batch_to_ids([b["input_words"] for b in batch])
    output_dict["target_tidx"] = torch.LongTensor(
        [b["target_tidx"] for b in batch]
    )

    for b in batch:
        output_dict["verb"].append(b["verb"])
        output_dict["frame"].append(b["frame"])
        output_dict["ex_idx"].append(b["ex_idx"])
    return output_dict


class RibeiroDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self._preprocess()

    def __len__(self):
        return len(self.out_inputs)

    def __getitem__(self, idx):
        return self.out_inputs[idx]

    def _preprocess(self):
        self.out_inputs = []
        for df_dict in self.df.to_dict("records"):
            inputs = {}
            inputs["input_words"] = df_dict["text_widx"].split()
            inputs["target_tidx"] = df_dict["target_widx"]
            inputs.update(
                {
                    "frame": df_dict["frame"],
                    "verb": df_dict["verb"],
                    "ex_idx": df_dict["ex_idx"],
                }
            )
            self.out_inputs.append(inputs)


class RibeiroEmbedding:
    def __init__(
        self, elmo_options_file, elmo_weights_file, batch_size, device
    ):
        self.model = Elmo(elmo_options_file, elmo_weights_file, 2, dropout=0)
        self.model = self.model.to(device).eval()
        self.batch_size = batch_size
        self.device = device

    def get_embedding(self, df):
        ds = RibeiroDataset(df)
        dl = DataLoader(
            ds,
            batch_size=self.batch_size,
            collate_fn=ribeiro_collate_fn,
            shuffle=False,
        )

        vec_list = []
        for batch in dl:
            with torch.no_grad():
                temp_outputs = self.model(batch["input_ids"].to(self.device))[
                    "elmo_representations"
                ]
                outputs = temp_outputs[0] + temp_outputs[1]
                embs = outputs[
                    torch.LongTensor(range(len(batch["target_tidx"]))),
                    batch["target_tidx"],
                ]
                vec_list += list(embs.cpu().detach().numpy())

        df_vec = (
            df.reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "vec_id"})
        )
        vec_array = np.array(vec_list)
        return df_vec, vec_array


class RibeiroClustering:
    def __init__(self):
        self.weighting = "top"
        self.seed = 1337
        self.iterations = 20

    def _clustering(self, vec_array):
        cossim_matrix = 1 - cosine_similarity(vec_array, vec_array)
        cossim_matrix *= np.triu(np.ones([len(vec_array), len(vec_array)]), 1)

        mean = cossim_matrix[cossim_matrix.nonzero()].mean()
        std = cossim_matrix[cossim_matrix.nonzero()].std()
        t = (mean + std) / 2
        t_cossim_matrix = np.where(cossim_matrix >= t, 0, cossim_matrix)

        i_list, j_list = np.nonzero(t_cossim_matrix)
        edge_list = [
            (i, j, {"weight": t_cossim_matrix[i, j]})
            for i, j in zip(i_list, j_list)
        ]

        g = nx.Graph()
        g.add_nodes_from(range(len(vec_array)))
        g.add_edges_from(edge_list)

        chinese_whispers(
            g,
            weighting=self.weighting,
            seed=self.seed,
            iterations=self.iterations,
        )
        cluster_dict = {}
        for i, (_, c) in enumerate(aggregate_clusters(g).items()):
            for cc in c:
                cluster_dict[cc] = {"label": i}

        cluster_array = [
            i + 1
            for i in list(pd.DataFrame(cluster_dict).T.sort_index()["label"])
        ]
        return cluster_array

    def step(self, df, vec_array):
        df["frame_cluster"] = self._clustering(vec_array)
        return df
