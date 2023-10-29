from fastcluster import linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist


class OnestepClustering:
    def __init__(self, clustering):
        self.clustering = clustering

    def make_params(self, df, vec_array):
        z = linkage(
            pdist(vec_array), method=self.clustering, preserve_input=False
        )
        params = {"th": z[-len(set(df["frame"])) + 1][2] + 1e-6}
        return params

    def _clustering(self, vec_array, params):
        z = linkage(
            pdist(vec_array), method=self.clustering, preserve_input=False
        )
        return fcluster(z, t=params["th"], criterion="distance")

    def step(self, df, vec_array, params):
        df["frame_cluster"] = self._clustering(vec_array, params)
        return df
