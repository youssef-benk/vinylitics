import pandas as pd
from datasets import load_dataset
from pathlib import Path



def basic_cleaning(sentence):
    if isinstance(sentence, str):
        return sentence.lower().strip()
    return sentence

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataset."""

    # Create a copy to avoid modifying the original dataset
    df_cleaned = df.copy()

    # Remove unnecessary columns
    drop_columns = ["Unnamed: 0", "track_id", "album_name", 'track_genre']
    df_cleaned.drop(columns=drop_columns, errors="ignore", inplace=True)

    # Drop missing values
    df_cleaned.dropna(inplace=True)

    # Remove invalid tempo and time_signature values
    df_cleaned = df_cleaned[(df_cleaned.tempo > 0) & (df_cleaned.time_signature > 0)]

    # Remove unpopular songs (popularity = 0)
    df_cleaned = df_cleaned[df_cleaned.popularity > 0]

    # Apply basic text cleaning to 'artists' and 'track_name' columns
    df_cleaned['track_name'] = df_cleaned['track_name'].apply(basic_cleaning)
    df_cleaned['artists'] = df_cleaned['artists'].apply(basic_cleaning)

    print("✅ data cleaned")

    return df_cleaned

def load_data(dataset_name="maharshipandya/spotify-tracks-dataset"):
    """Loads the dataset from Hugging Face."""
    # Load dataset
    ds = load_dataset(dataset_name)
    df = ds['train'].to_pandas()

    return df

def get_data_with_cache(
    ds_link: str,
    cache_path:Path) -> pd.DataFrame:
    """Loads the dataset from Hugging Face or from local cache if the file exists."""
    if cache_path.is_file():
        print("📚 Loading data from cache")
        df = pd.read_csv(cache_path)

    else:
        print("🚀 Loading data from Hugging Face")
        df = load_data(ds_link)
        df.to_csv(cache_path.joinpath(f"{ds_link}.csv"), index=True)

    return df
