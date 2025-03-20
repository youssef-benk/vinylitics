from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from vinylitics.preproc.recommender import recommend_track

app = FastAPI()

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
    result = recommend_track(track_name, artist)
    if result is not None:
        return {'result': result.to_dict()}
    else:
        return {"error": "Track not found"}
