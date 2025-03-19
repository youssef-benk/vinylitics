from vinylitics.params import *
from vinylitics.preproc.data import get_data_with_cache, clean_data, load_data
from vinylitics.preproc.preprocessor import preprocess_features, fit_preprocessor
from vinylitics.preproc.model import neighbors_fit
from colorama import Fore, Style
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pathlib import Path


def preprocess(ds="maharshipandya/spotify-tracks-dataset"):
    """
    - Query the spotify dataset from HuggingFace
        maharshipandya/spotify-tracks-dataset
    - Preprocess the data
    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    data_cache_path = Path(LOCAL_DATA_PATH).joinpath(f"{ds}.csv")

    X = get_data_with_cache(ds, data_cache_path)
    X_clean = clean_data(X)

    if not "preproc.dill" in os.listdir():
         X_preproc = fit_preprocessor(X_clean)
    else:
        X_preproc = preprocess_features(X_clean)

    return X_preproc


def train(X:pd.DataFrame, n_neighbors:int=N_NEIGHBORS, algorithm:str='brute', metrics:str='cosine'):
    """
    - Train the nearest neighbors model
    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: train" + Style.RESET_ALL)

    neighbors_fit(X_preproc, n_neighbors=n_neighbors, algorithm=algorithm, metrics=metrics)

    print("✅ train() done \n")
    return None


if __name__ == '__main__':
    X_preproc= preprocess()
    train(X_preproc)
