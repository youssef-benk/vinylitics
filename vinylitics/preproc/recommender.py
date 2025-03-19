from vinylitics.preproc.data import load_data
from vinylitics.preproc.preprocessor import preprocess_features
from vinylitics.preproc.model import find_neighbors
import pandas as pd
from vinylitics.params import *
from pathlib import Path


def recommend_track(track_name, artist):
    track_name = track_name.lower().strip()
    artist = artist.lower().strip()
    df = load_data()

    try:
        selected_track = df[(df['track_name'] == track_name) & (df['artists'] == artist)]
    except:
        print("Track not found")
        return None

    track_preproc = preprocess_features(selected_track)

    find_neighbors(track_preproc)
    distances, indices = find_neighbors(track_preproc)

    result = df.iloc[indices[0][1:]].sort_values(by='popularity', ascending=False)
    print(result)
    return result

# recommend_track("Shape of You", "Ed Sheeran")
