import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from vinylitics.params import *
from vinylitics.preproc import data
import dill
from colorama import Fore, Style




def fit_preprocessor(X:pd.DataFrame):
    """
    Fits the preprocessing pipeline to the provided DataFrame.

    Args:
        X (pd.DataFrame): The input DataFrame containing features to preprocess.

    Returns:
        Pipeline: A fitted preprocessing pipeline.
    """

    # select feature groups
    continuous_features = ['popularity', 'duration_ms', 'loudness', 'tempo']
    bounded_features = [
        'danceability', 'energy', 'speechiness'
        , 'acousticness', 'instrumentalness', 'liveness'
        , 'valence', 'genre_danceability', 'genre_energy'
        , 'genre_valence'
        ]
    categorical_features = ['time_signature', 'mode']

    X = X[KEEP_COLUMNS]

    def create_sklearn_preprocessor():
        '''Scikit-learn pipeline that transforms a cleaned dataset of shape (_, 7)
        into a preprocessed one of fixed shape (_, 28).'''

        continuous_preproc = Pipeline([
            ("continuous_scaler", RobustScaler())
            ])

        bounded_preproc = Pipeline([
            ("bounded_scaler", MinMaxScaler())
            ])

        categorical_preproc = Pipeline([
            ("ohe", OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False))])

        preproc = make_column_transformer(
            (continuous_preproc, continuous_features)
            , (bounded_preproc, bounded_features)
            , (categorical_preproc, categorical_features)
            , remainder='passthrough'
        )

        preproc_pipe = make_pipeline(preproc)

        return preproc_pipe

    preproc_pipe = create_sklearn_preprocessor()
    X_preproc = preproc_pipe.fit_transform(X)
    print("✅ Preprocessing pipeline fitted and X_preprocessed, with shape", X_preproc.shape)

    # Save transformer to a file using dill from your notebook
    with open("preproc_pipe.dill", "wb") as f:
        dill.dump(preproc_pipe, f)

    return preproc_pipe, X_preproc


def preprocess_features(X:pd.DataFrame):
    '''
        Preprocess the features from the input dataframe.
    '''

    X = X[KEEP_COLUMNS]

    if not "preproc_pipe.dill" in os.listdir():
        print(Fore.RED + "❌ The preprocessing pipeline has not been fitted yet. Please run fit_preprocessor first." + Style.RESET_ALL)
        return None

    with open("preproc_pipe.dill", "rb") as f:
        preproc_pipe = dill.load(f)

    X_preproc = preproc_pipe.transform(X)
    print("✅ X_preprocessed, with shape", X_preproc.shape)

    return X_preproc
