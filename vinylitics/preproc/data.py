import pandas as pd
from datasets import load_dataset
from pathlib import Path
from vinylitics.params import LOCAL_DATA_PATH



def basic_cleaning(sentence):
    if isinstance(sentence, str):
        return sentence.lower().strip()
    return sentence

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataset."""

    # Create a copy to avoid modifying the original dataset
    df_cleaned = df.copy()

    # Remove unnecessary columns
    drop_columns = ["Unnamed: 0", "track_id", "album_name", 'track_genre', 'year', 'Unnamed: 0.1']
    df_cleaned.drop(columns=drop_columns, errors="ignore", inplace=True)

    # Drop missing values
    df_cleaned.dropna(inplace=True)

    # Remove invalid tempo and time_signature values
    df_cleaned = df_cleaned[(df_cleaned.tempo > 0)]

    # Remove unpopular songs (popularity = 0)
    df_cleaned = df_cleaned[df_cleaned.popularity > 0]

    # Apply basic text cleaning to 'artists' and 'track_name' columns
    df_cleaned['track_name'] = df_cleaned['track_name'].apply(basic_cleaning)
    df_cleaned['artists'] = df_cleaned['artists'].apply(basic_cleaning)

    print("✅ data cleaned")

    return df_cleaned

def load_data(dataset_name="dataframe_2") -> pd.DataFrame:
    """Loads the dataset from the given path."""
    # Load dataset
    data_path = Path(LOCAL_DATA_PATH).joinpath(f"{dataset_name}.csv")
    df = pd.read_csv(data_path)
    print("✅ data loaded")
    return df
