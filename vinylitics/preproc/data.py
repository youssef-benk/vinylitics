import pandas as pd
from datasets import load_dataset



def basic_cleaning(sentence):
    if isinstance(sentence, str):
        return sentence.lower().strip()
    return sentence

def load_and_clean_data(dataset_name):
    """Loads the dataset from Hugging Face and cleans it."""
    # Load dataset
    ds = load_dataset(dataset_name)
    df = ds['train'].to_pandas()

    # Create a copy to avoid modifying the original dataset
    df_cleaned = df.copy()

    # Remove unnecessary columns
    drop_columns = ["Unnamed: 0", "track_id", "album_name"]
    df_cleaned.drop(columns=[col for col in drop_columns if col in df_cleaned.columns], errors="ignore", inplace=True)

    # Drop missing values
    df_cleaned.dropna(inplace=True)

    # Remove invalid tempo and time_signature values
    df_cleaned = df_cleaned[(df_cleaned.tempo > 0) & (df_cleaned.time_signature > 0)]

    # Remove unpopular songs (popularity = 0)
    df_cleaned = df_cleaned[df_cleaned.popularity > 0]

    # Apply basic text cleaning to 'artists' and 'track_name' columns
    df_cleaned['track_name'] = df_cleaned['track_name'].map(lambda x: basic_cleaning(x))
    df_cleaned['artists'] = df_cleaned['artists'].map(lambda x: basic_cleaning(x))

    print("✅ data cleaned")

    return df_cleaned

#comment
