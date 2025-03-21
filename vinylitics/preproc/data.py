import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path
from vinylitics.params import LOCAL_DATA_PATH



def basic_cleaning(sentence):
    ############################################
    # DO WE STILL NEED THIS AFTER RECOMMENDER? #
    ############################################
    if isinstance(sentence, str):
        return sentence.lower().strip().strip('#')
    return sentence

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataset."""

    # Create a copy to avoid modifying the original dataset
    df_cleaned = df.copy()

    # Remove unnecessary columns
    drop_columns = ['Unnamed: 0.1', 'Unnamed: 0', 'source','album_name', 'explicit']
    df_cleaned.drop(columns=drop_columns, errors="ignore", inplace=True)

    # Remove invalid tempo and time_signature values
    df_cleaned = df_cleaned[(df_cleaned.tempo > 0)]

    # Remove unpopular songs (popularity = 0)
    df_cleaned = df_cleaned[df_cleaned.popularity > 0]

    # Fill missing numeric values with median
    df_cleaned['duration_ms'].fillna(df_cleaned['duration_ms'].median(), inplace=True)
    df_cleaned['year'].fillna(df_cleaned['year'].median(), inplace=True)

    # Fill categorical values with most frequent (mode)
    df_cleaned['time_signature'].fillna(df_cleaned['time_signature'].mode()[0], inplace=True)

    # Fill missing track_genre with "Unknown"
    df_cleaned['track_genre'].fillna("Unknown", inplace=True)

    #########################################################################
    ### SIMPLER 'ONE HOT ENCODING' FOR GENRE                              ###
    #########################################################################

    # Create a copy to avoid modifying the original dataset
    df_cleaned = df.copy()

    # Compute mean values per genre for selected features
    genre_features = df_cleaned.groupby('track_genre')[['danceability', 'energy', 'valence']].mean()

    # Rename columns to prevent merge conflicts
    genre_features.rename(columns={
        'danceability': 'genre_danceability',
        'energy': 'genre_energy',
        'valence': 'genre_valence'
    }, inplace=True)

    # Merge genre-based feature averages back into the dataset
    df_cleaned = df_cleaned.merge(genre_features, on='track_genre', how='left')

    # Drop original 'track_genre' and the original numeric columns
    df_cleaned.drop(columns=['track_genre'], inplace=True)

    #########################################################################
    ### CAPPING DURATION TO REMOVE OUTLIERS                               ###
    #########################################################################

    # Cap 'duration_ms' at 99th percentile because numerous outliers
    duration_cap = np.percentile(df_cleaned['duration_ms'], 99)
    df_cleaned['duration_ms'] = np.where(df_cleaned['duration_ms'] > duration_cap, duration_cap, df_cleaned['duration_ms'])

    #########################################################################
    ### USING COS AND SINE FOR KEY (CYCLICAL)                             ###
    #########################################################################

    # transform 'key' (cyclical feature) into sine and cosine components and drop original one

    df_cleaned['key_sin'] = np.sin(2 * np.pi * df_cleaned['key'] / 12)
    df_cleaned['key_cos'] = np.cos(2 * np.pi * df_cleaned['key'] / 12)
    df_cleaned.drop(columns=['key'], inplace=True)



    # Apply basic text cleaning to 'artists' and 'track_name' columns
    ############################################
    # DO WE STILL NEED THIS AFTER RECOMMENDER? #
    ############################################
    df_cleaned['track_name'] = df_cleaned['track_name'].apply(basic_cleaning)
    df_cleaned['artists'] = df_cleaned['artists'].apply(basic_cleaning)

    print("🧹 data cleaned, with shape ", df_cleaned.shape)

    return df_cleaned





def load_data(dataset_name="dataframe_2") -> pd.DataFrame:
    """Loads the dataset from the given path."""
    # Load dataset
    data_path = Path(LOCAL_DATA_PATH).joinpath(f"{dataset_name}.csv")
    df = pd.read_csv(data_path)
    print("🚀 data loaded, with shape ", df.shape)
    return df
