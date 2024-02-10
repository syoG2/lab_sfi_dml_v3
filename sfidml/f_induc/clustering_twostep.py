from collections import Counter

import numpy as np
import pandas as pd
from fastcluster import linkage
from pyclustering.cluster.xmeans import kmeans_plusplus_initializer, xmeans
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist
from scipy.special import comb
from sklearn.metrics import confusion_matrix


class TwostepClustering:
    def __init__(self, clustering_method1, clustering_method2):
        self.clustering_method1 = clustering_method1
        self.clustering_method2 = clustering_method2

    def _make_vec_1st(self, df, vec_array, verb):
        df_verb = df[df["verb"] == verb].copy()
        verb_vec_array = vec_array[df_verb["vec_id"]]
        return df_verb, verb_vec_array

    def _clustering_1st(self, vec_array, params):
        if self.clustering_method1 == "average":
            if len(vec_array) >= 2:
                z = linkage(
                    pdist(vec_array),
                    method="average",
                    metric="euclidean",
                    preserve_input=False,
                )
                cluster_array = fcluster(z, t=params["lth"], criterion="distance")
            else:
                cluster_array = np.array([1])
        elif self.clustering_method1 == "xmeans":
            init_center = kmeans_plusplus_initializer(vec_array, 1).initialize()
            xm = xmeans(
                vec_array,
                init_center,
                ccore=False,
                kmax=params["kmax"],
                random_state=0,
            )
            xm.process()
            cluster_array = np.array([-1] * len(vec_array))
            for idx, clusters in enumerate(xm.get_clusters()):
                for sent_idx in clusters:
                    cluster_array[sent_idx] = idx + 1
        elif self.clustering_method1 == "1cpv":
            cluster_array = np.array([1] * len(vec_array))
        return cluster_array

    def _make_vec_2nd(self, df, vec_array, count):
        vec_list, df_list = [], []
        for plu in sorted(set(df["plu_local"])):
            df_cluster = df[df["plu_local"] == plu].copy()
            vec_list.append(
                np.average(
                    [vec_array[vec_id] for vec_id in df_cluster["vec_id"]],
                    axis=0,
                )
            )
            df_cluster.loc[:, "plu_global"] = count + 1
            df_list.append(df_cluster)
            count += 1
        return vec_list, pd.concat(df_list, axis=0)

    def _clustering_2nd(self, vec_array, params):
        z = linkage(
            pdist(vec_array),
            method=self.clustering_method2,
            preserve_input=False,
        )
        cluster_array = fcluster(
            z, t=self._decide_t(z, params["gth"]), criterion="distance"
        )
        return cluster_array

    def _decide_t(self, z, gth):
        n_sides, n_points = 0, len(z) + 1
        for i in range(len(z)):
            if n_points <= z[i, 0]:
                pre_n_points0 = int(z[int(z[i, 0] - n_points), 3])
                pre_n_sides0 = comb(pre_n_points0, 2, exact=True)
                n_sides -= pre_n_sides0
            if n_points <= z[i, 1]:
                pre_n_points1 = int(z[int(z[i, 1] - n_points), 3])
                pre_n_sides1 = comb(pre_n_points1, 2, exact=True)
                n_sides -= pre_n_sides1
            n_sides += comb(int(z[i, 3]), 2, exact=True)
            probs = n_sides / comb(n_points, 2, exact=True)
            if probs >= gth:
                t = z[i, 2]
                break
        return t

    def make_confusion_matrix(self, df):
        true = df.groupby("verb").nunique()["frame"].values
        pred = df.groupby("verb").max()["plu_local"].values
        labels = [str(i) for i in range(1, max([max(true), max(pred)]))]
        cm = confusion_matrix(true.astype(str), pred.astype(str), labels=labels)
        return pd.DataFrame(cm, index=labels, columns=labels)

    def make_params(self, df, vec_array):
        params = {}
        if self.clustering_method1 == "average":
            lth_list = []
            for verb in sorted(set(df["verb"])):
                df_verb = df[df["verb"] == verb]
                verb_vec_array = vec_array[df_verb["vec_id"]]
                lth_dict = {
                    "verb": verb,
                    "n_frames": len(set(df_verb["frame"])),
                    "n_texts": len(verb_vec_array),
                }
                if len(verb_vec_array) >= 2:
                    z = linkage(
                        pdist(verb_vec_array),
                        method=self.clustering_method1,
                        metric="euclidean",
                        preserve_input=False,
                    )
                    for _, _, lth, _ in z:
                        lth_dict = lth_dict.copy()
                        lth_dict["lth"] = lth
                        lth_list.append(lth_dict)
                else:
                    lth_dict["lth"] = 0
                    lth_list.append(lth_dict)

            df_lth = pd.DataFrame(lth_list).sort_values("lth", ascending=False)
            params["lth"] = (
                df_lth["lth"][
                    : len(set(df["verb_frame"])) - len(set(df["verb"])) + 1
                ].values[-1]
                + 1e-6
            )
        elif self.clustering_method1 == "xmeans":
            params["kmax"] = max(
                df.groupby("verb").agg(set)["frame"].apply(lambda x: len(x))
            )

        vf2f = {vf: f for vf, f in zip(df["verb_frame"], df["frame"])}
        params["gth"] = sum(
            [comb(i, 2, exact=True) for i in Counter(vf2f.values()).values()]
        ) / comb(len(vf2f.values()), 2, exact=True)
        return params

    def step(self, df, vec_array1, vec_array2, params):
        vec_array_2nd, df_2nd_list = [], []
        for verb in sorted(set(df["verb"])):
            df_1st, vec_array_1st = self._make_vec_1st(df, vec_array1, verb)
            df_1st.loc[:, "plu_local"] = self._clustering_1st(vec_array_1st, params)
            _vec_array_2nd, _df_2nd = self._make_vec_2nd(
                df_1st, vec_array2, len(vec_array_2nd)
            )
            vec_array_2nd += _vec_array_2nd
            df_2nd_list.append(_df_2nd)
        vec_array_2nd = np.array(vec_array_2nd)
        df_2nd = pd.concat(df_2nd_list, axis=0)

        map_1to2 = {
            c1 + 1: c2
            for c1, c2 in enumerate(self._clustering_2nd(vec_array_2nd, params))
        }
        df_2nd["frame_cluster"] = df_2nd["plu_global"].map(map_1to2)
        return df_2nd
