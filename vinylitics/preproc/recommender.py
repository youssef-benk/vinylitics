from vinylitics.preproc.data import load_and_clean_data
from vinylitics.preproc.preprocessor import preprocess_features
from vinylitics.preproc.model import neighbors_fit, find_neighbors

df = load_and_clean_data()

def recommend_track(track_name, artist):
    track_name = track_name.lower().strip()
    artist = artist.lower().strip()
    try:
        selected_track = df[(df['track_name'] == track_name) & (df['artists'] == artist)]
    except:
        print("Track not found")
        return None

    track_preproc = preprocess_features(selected_track)

    df_preproc = preprocess_features(df)
    pca, model = neighbors_fit(df_preproc)

    neighbors = find_neighbors(track_preproc, model, pca)
    print(neighbors)

    return None

recommend_track("Shape of You", "Ed Sheeran")
