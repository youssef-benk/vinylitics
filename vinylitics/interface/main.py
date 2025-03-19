from vinylitics.params import *
from vinylitics.preproc.data import load_and_clean_data
from vinylitics.preproc.preprocessor import preprocess_features
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
    X_preproc = preprocess_features(X)

    return X_preproc





if __name__ == '__main__':
    preprocess()
