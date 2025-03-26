from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from vinylitics.preproc.recommender import recommend_track
from vinylitics.preproc.getmp3 import  get_mp3
from vinylitics.preproc.translator import extract_low_features_from_mp3, predict_high_features, get_ordered_columns
from thefuzz import process, fuzz
from vinylitics.preproc.data import load_data, clean_data
from vinylitics.params import GCP_PROJECT, BQ_DATASET, DS

app = FastAPI()

query_raw = f"""
    SELECT *
    FROM `{GCP_PROJECT}`.{BQ_DATASET}.{DS}_raw
    """

query_cleaned = f"""
    SELECT *
    FROM `{GCP_PROJECT}`.{BQ_DATASET}.{DS}_cleaned
    """

app.state.df_og = load_data(gcp_project=GCP_PROJECT, query=query_raw, dataset_name='dataframe_2')
app.state.df = clean_data(app.state.df_og)

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Allows all origins
    allow_credentials = True,
    allow_methods=['*'],  # Allows all methods
    allow_headers=['*'],  # Allows all headers
)

@app.get('/')
def root():
    response = {
        'greeting': 'Welcome to Vynilitics API',
    }

    return response

@app.get('/fuzzy_search')
def fuzzy_search(track_name: str, artist: str):
    track_name = track_name.lower().strip()
    artist = artist.lower().strip()

    selected_track = app.state.df[(app.state.df['track_name'] == track_name) & (app.state.df['artists'] == artist)]
    if selected_track.empty:
        # Deploy fuzzy search if track not found
        # Combine track name and artist for fuzzy matching
        query = track_name + " " + artist
        # Create a list of combined strings for each track in the dataframe
        choices = (app.state.df_og['track_name'] + " by " + app.state.df_og['artists']).tolist()
        # Get the top 3 matches
        return {"result": "No exact match found", "choices": process.extract(query, choices, limit=3, scorer=fuzz.token_set_ratio)}
    else:
        return {"result": "Exact match found", "track": selected_track.to_dict()}



@app.get("/predict_spotify")
def predict_spotify(track_name: str, artist: str):
    """_summary_

    Args:
        track_name (str): _description_
        artist (str): _description_

    Returns:
        Hidden gems: similar but less popular songs from our Spotify dataset.
    """
    # Call the recommender function
    result, selected_track = recommend_track(track_name, artist, app.state.df)
    if result is not None:
        return {'result': result.to_dict(),
                'sel_track': selected_track.to_dict()}
    else:
        return {"error": "Track not found"}


@app.get("/song_scrap")
def song_scrap(track_name: str):
    """_summary_

    Args:
        track_name (str): _description_
    """
    ordered_cols = get_ordered_columns()
    audio_path = get_mp3(track_name)

    # Compute the low features with librosa and gets there stat
    low_features = extract_low_features_from_mp3(audio_path)

    # Compute the high features with our own trained RNN
    df_high = predict_high_features(low_features)

    if df_high is not None:
        return {'result': df_high.to_dict()}
    else:
        return {"error": "Something went wrong."}
