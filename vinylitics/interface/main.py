from vinylitics.params import *
from vinylitics.preproc.data import clean_data, load_data
from vinylitics.preproc.preprocessor import preprocess_features, fit_preprocessor
from vinylitics.preproc.model import neighbors_fit
from colorama import Fore, Style
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pathlib import Path


def preprocess(ds="dataframe_2"):
    """
    - Query the spotify dataset from HuggingFace
        maharshipandya/spotify-tracks-dataset
    - Preprocess the data
    """
    print(Fore.MAGENTA + "\n :star:️ Use case: preprocess" + Style.RESET_ALL)
    X = load_data(ds)
    X_clean = clean_data(X)

    if not "preproc.dill" in os.listdir():
        preproc_pipe, X_preproc = fit_preprocessor(X_clean)
    else:
        X_preproc = preprocess_features(X_clean)
    return X_preproc

def train(X:pd.DataFrame, n_neighbors:int=N_NEIGHBORS, algorithm:str='brute', metrics:str='cosine'):
    """
    - Train the nearest neighbors model
    """
    print(Fore.MAGENTA + "\n :star:️ Use case: train" + Style.RESET_ALL)
    neighbors_fit(X, n_neighbors=n_neighbors, algorithm=algorithm, metrics=metrics)
    print("✅ train() done \n")
    return None


if __name__ == '__main__':
    X_preproc= preprocess()
    train(X_preproc, n_neighbors=10, algorithm='brute', metrics='cosine')
    train(X_preproc)
