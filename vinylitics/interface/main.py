from vinylitics.params import *
from vinylitics.preproc.data import clean_data, load_data, load_data_to_bq
from vinylitics.preproc.preprocessor import preprocess_features, fit_preprocessor
from vinylitics.preproc.model import neighbors_fit
from colorama import Fore, Style
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from vinylitics.preproc.recommender import recommend_track


def preprocess(ds="dataframe_2", load_to_bq=False):
    """
    - Query the spotify dataset from HuggingFace
        maharshipandya/spotify-tracks-dataset
    - Preprocess the data
    """
    print(Fore.MAGENTA + "\n ⭐️Use case: preprocess" + Style.RESET_ALL)

    query_raw = f"""
    SELECT *
    FROM `{GCP_PROJECT}`.{BQ_DATASET}.{ds}_raw
    """
    X = load_data(gcp_project=GCP_PROJECT, query=query_raw, dataset_name=ds)

    # Clean the data
    X_clean = clean_data(X)

    # Omly load to BQ if the flag is set to True
    if load_to_bq:
        # Save the raw data to BigQuery
        load_data_to_bq(
            data=X,
            gcp_project=GCP_PROJECT,
            bq_dataset=BQ_DATASET,
            table=f"{ds}_raw",
            truncate=True
        )

        # Save the cleaned data to BigQuery
        load_data_to_bq(
            data=X_clean,
            gcp_project=GCP_PROJECT,
            bq_dataset=BQ_DATASET,
            table=f"{ds}_cleaned",
            truncate=True
        )
    if not "preproc.dill" in os.listdir():
        preproc_pipe, X_preproc = fit_preprocessor(X_clean)
    else:
        X_preproc = preprocess_features(X_clean)
    return X_preproc

def train(X:pd.DataFrame, n_neighbors:int=N_NEIGHBORS, algorithm:str='KD', metrics:str='cosine'):
    """
    - Train the nearest neighbors model
    """
    print(Fore.MAGENTA + "\n ⭐️Use case: train" + Style.RESET_ALL)
    neighbors_fit(X, n_neighbors=n_neighbors, algorithm=algorithm, metrics=metrics)
    print("✅ train() done \n")
    return None


if __name__ == '__main__':
    # X_preproc= preprocess()
    # train(X_preproc, n_neighbors=10, algorithm='auto', metrics='cosine')
    recommend_track("Do I wanna know", "Arctic Monkeys", clean_data(pd.read_csv('./raw_data/dataframe_2.csv')))
