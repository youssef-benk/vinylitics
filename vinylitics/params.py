import os
from pathlib import Path

##################  CONSTANTS  #####################
LOCAL_DOCKER_PATH = Path("raw_data")
LOCAL_DATA_PATH =  os.path.join(os.path.expanduser('~'), "code", "youssef-benk", "vinylitics", "raw_data")

KEEP_COLUMNS = [
      'tempo',
    'danceability', 'energy', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence'
    ] #'loudness', 'genre_danceability', 'genre_energy', 'genre_valence','mode','key_sin', 'key_cos', 'popularity', 'duration_ms', ,  'time_signature'

##################  VARIABLES  ##################
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_PROJECT_WAGON = os.environ.get("GCP_PROJECT_WAGON")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
GAR_IMAGE = os.environ.get("GAR_IMAGE")
GAR_MEMORY = os.environ.get("GAR_MEMORY")

N_NEIGHBORS = 10
COLUMN_NAMES_RAW = ['track_name', 'artists', 'track_id',
       'album_name', 'popularity', 'duration_ms', 'explicit', 'danceability',
       'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
       'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature',
       'track_genre', 'source', 'year']
DS = "dataframe_2"
