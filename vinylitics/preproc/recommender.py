from vinylitics.preproc.data import load_and_clean_data
from vinylitics.preproc.preprocessor import preprocess_features
from vinylitics.preproc.model import neighbors_fit, find_neighbors
import pandas as pd
from vinylitics.params import *
from pathlib import Path

df = load_and_clean_data()

def recommend_track(track_name, artist):
    track_name = track_name.lower().strip()
    artist = artist.lower().strip()
    data_cache_path = Path(LOCAL_DATA_PATH).joinpath(f"{ds}.csv")

    if not data_cache_path.is_file():
        print("❌ Training data not found")
        return None
    df = pd.read_csv(data_cache_path)
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
