import numpy as np
from sklearn.manifold import TSNE


def project_embedding(df, vec_array, random_state=0):
    df = df.copy()
    if len(vec_array) >= 2:
        tsne = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            random_state=random_state,
        )
        df.loc[:, ["x", "y"]] = tsne.fit_transform(vec_array)
    else:
        df.loc[:, ["x", "y"]] = np.array([0, 0])
    return df
