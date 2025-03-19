from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.decomposition import PCA

def neighbors_fit(
    X:pd.DataFrame,
    n_neighbors:int=5,
    algorithm:str='ball_tree',
    metrics:str='cosine'):
    """Fit the nearest neighbors on the preprocessd data

    Args:
        X (pd.DataFrame): preprocessed data
        n_neighbors (int, optional): _description_. Defaults to 5.
        algorithm (str, optional): _description_. Defaults to 'ball_tree'.
        metrics (str, optional): _description_. Defaults to 'cosine'.
    """
    pca = PCA()
    X_proj = pca.fit_transform(X)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metrics).fit(X_proj)

    return pca, nbrs

def find_neighbors(X_pred:pd.DataFrame, model, pca):
    """Find the neighbors of a given song

    Args:
        X_pred (pd.DataFrame): The song to find neighbors for
        model (str, optional): Nearest neighbors model.
        pca (PCA, optional): fitted pca
    """
    X_pred_proj = pca.transform(X_pred)

    distances, indices = model.kneighbors(X_pred_proj)

    return distances, indices
