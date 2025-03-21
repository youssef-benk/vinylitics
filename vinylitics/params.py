import os

##################  CONSTANTS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "youssef-benk", "vinylitics", "raw_data")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "code", "youssef-benk", "vinylitics", "raw_data")

KEEP_COLUMNS = [
    'popularity', 'duration_ms', 'loudness', 'tempo', 'danceability'
    , 'energy', 'speechiness', 'acousticness', 'instrumentalness'
    , 'liveness', 'valence', 'genre_danceability', 'genre_energy'
    , 'genre_valence', 'key_sin', 'key_cos', 'time_signature', 'mode'
]

SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET")

N_NEIGHBORS = 10
