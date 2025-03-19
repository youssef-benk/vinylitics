from vinylitics.params import *
from vinylitics.preproc.data import load_and_clean_data
from vinylitics.preproc.preprocessor import preprocess_features, fit_preprocessor
from vinylitics.preproc.model import neighbors_fit
from colorama import Fore, Style
import pandas as pd
from sklearn.neighbors import NearestNeighbors



def preprocess(ds="maharshipandya/spotify-tracks-dataset"):
    """
    - Query the spotify dataset from HuggingFace
        maharshipandya/spotify-tracks-dataset
    - Preprocess the data
    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    X = load_and_clean_data(ds)

    if ds == "maharshipandya/spotify-tracks-dataset":
        preproc_pipe, X_preproc = fit_preprocessor(X)
    else:
        X_preproc = preprocess_features(X)

    return X_preproc


def train(X:pd.DataFrame, n_neighbors:int=5, algorithm:str='brute', metrics:str='cosine'):
    """
    - Train the nearest neighbors model
    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: train" + Style.RESET_ALL)

    neighbors = neighbors_fit(X_preproc, n_neighbors=n_neighbors, algorithm=algorithm, metrics=metrics)

    print("✅ train() done \n")
    return None


if __name__ == '__main__':
    X_preproc= preprocess()
    train(X_preproc)
