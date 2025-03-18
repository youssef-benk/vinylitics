import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from vinylitics.params import *
from vinylitics.preproc import data

def preprocess_features(X:pd.DataFrame):
    '''
        Create a preprocessing pipeline
    '''
    # Define the columns to keep
    cat_col = ['key', 'mode', 'time_signature']

    num_col = ['duration_ms', 'danceability', 'energy','loudness',
       'speechiness', 'acousticness', 'instrumentalness', 'liveness',
       'valence', 'tempo']

    X = X[KEEP_COLUMNS]


    def create_sklearn_preprocessor():
        '''Scikit-learn pipeline that transforms a cleaned dataset of shape (_, 7)
        into a preprocessed one of fixed shape (_, 28).'''
        num_preproc = Pipeline([
            ("scaler", RobustScaler())])
        cat_preproc = Pipeline([
            ("ohe", OneHotEncoder(handle_unknown="ignore",
                                  sparse_output=False))])
        preproc = make_column_transformer(
            (num_preproc, num_col),
            (cat_preproc, cat_col),
            remainder='drop'
        )
        preproc_pipe = make_pipeline(preproc)
        return preproc_pipe
    preproc_pipe = create_sklearn_preprocessor()
    X_preproc = preproc_pipe.fit_transform(X)
    print("✅ X_preprocessed, with shape", X_preproc.shape)

    return X_preproc

