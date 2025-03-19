from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.decomposition import PCA
import dill
from sklearn.pipeline import make_pipeline

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
    pca.fit(X)
    X_proj = pca.transform(X)
    print("✅ Fitted PCA model")

        # Save transformer to a file using dill from your notebook
    with open("pca.dill", "wb") as f:
        dill.dump(pca, f)
    print("✅ Saved fitted PCA to pca.dill")

    neighbors = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metrics).fit(X_proj)
    print("✅ Fitted NearestNeighbors model")

    # Save transformer to a file using dill from your notebook
    with open("neighbors.dill", "wb") as f:
        dill.dump(neighbors, f)
    print("✅ Saved NearestNeighbors model to neighbors_pipe.dill")

    return None

def find_neighbors(X_pred:pd.DataFrame):
    """Find the neighbors of a given song

    Args:
        X_pred (pd.DataFrame): The song to find neighbors for
        model (str, optional): Nearest neighbors model.
        pca (PCA, optional): fitted pca
    """
    # Load pca from the file using dill
    with open("pca.dill", "rb") as f:
        pca = dill.load(f)

    # Load model from the file using dill
    with open("neighbors.dill", "rb") as f:
        neighbors = dill.load(f)

    X_proj = pca.transform(X_pred)
    print("✅ Transformed X_pred with PCA")

    distances, indices = neighbors.kneighbors(X_proj)
    print("✅ Found neighbors")
    return distances, indices

