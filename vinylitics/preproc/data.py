import pandas as pd
from datasets import load_dataset
from pathlib import Path
from vinylitics.params import LOCAL_DATA_PATH, COLUMN_NAMES_RAW
from colorama import Fore, Style
from google.cloud import bigquery




def basic_cleaning(sentence):
    if isinstance(sentence, str):
        return sentence.lower().strip().strip('#')
    return sentence

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataset."""

    # Create a copy to avoid modifying the original dataset
    df_cleaned = df.copy()

    # Remove unnecessary columns
    drop_columns = ["album_name", 'track_genre', 'year']
    df_cleaned.drop(columns=drop_columns, errors="ignore", inplace=True)

    # Remove invalid tempo and time_signature values
    df_cleaned = df_cleaned[(df_cleaned.tempo > 0)]

    # Remove unpopular songs (popularity = 0)
    df_cleaned = df_cleaned[df_cleaned.popularity > 0]

    # Apply basic text cleaning to 'artists' and 'track_name' columns
    df_cleaned['track_name'] = df_cleaned['track_name'].apply(basic_cleaning)
    df_cleaned['artists'] = df_cleaned['artists'].apply(basic_cleaning)

    print("🧹 data cleaned, with shape ", df_cleaned.shape)

    return df_cleaned

def load_data(gcp_project:str,
              query:str,
              dataset_name:str = "dataframe_2") -> pd.DataFrame:
    """Loads the dataset from the given path."""
    # Load dataset
    data_path = Path(LOCAL_DATA_PATH).joinpath(f"{dataset_name}.csv")

    if data_path.is_file():
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
        df = pd.read_csv(data_path)
        df = df[COLUMN_NAMES_RAW]
    else:
        print(Fore.BLUE + "\nLoad data from BigQuery..." + Style.RESET_ALL)
        client = bigquery.Client(project=gcp_project)
        query_job = client.query(query)
        result = query_job.result()
        df = result.to_dataframe()

        # Store as CSV if the BQ query returned at least one valid line
        if df.shape[0] > 1:
            df.to_csv(data_path, header=True, index=False)
    print("🚀 data loaded, with shape ", df.shape)
    return df

def load_data_to_bq(
        data: pd.DataFrame,
        gcp_project:str,
        bq_dataset:str,
        truncate: bool,
        table: str = "dataframe_2"
    ) -> None:
    """
    - Save the DataFrame to BigQuery
    - Empty the table beforehand if `truncate` is True, append otherwise
    """

    assert isinstance(data, pd.DataFrame)
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
    print(Fore.BLUE + f"\nSave data to BigQuery @ {full_table_name}...:" + Style.RESET_ALL)

    # Load data onto full_table_name

    # 🎯 HINT for "*** TypeError: expected bytes, int found":
    # After preprocessing the data, your original column names are gone (print it to check),
    # so ensure that your column names are *strings* that start with either
    # a *letter* or an *underscore*, as BQ does not accept anything else

    # TODO: simplify this solution if possible, but students may very well choose another way to do it
    # We don't test directly against their own BQ tables, but only the result of their query
    data.columns = [f"_{column}" if not str(column)[0].isalpha() and not str(column)[0] == "_" else str(column) for column in data.columns]

    client = bigquery.Client()

    # Define write mode and schema
    write_mode = "WRITE_TRUNCATE" if truncate else "WRITE_APPEND"
    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)

    print(f"\n{'Write' if truncate else 'Append'} {full_table_name} ({data.shape[0]} rows)")

    # Load data
    job = client.load_table_from_dataframe(data, full_table_name, job_config=job_config)
    result = job.result()  # wait for the job to complete

    print(f"✅ Data saved to bigquery, with shape {data.shape}")
