from vinylitics.preproc.data import load_data, clean_data
from vinylitics.preproc.preprocessor import preprocess_features
from vinylitics.preproc.model import find_neighbors
from vinylitics.params import *
from thefuzz import process, fuzz
import pandas as pd
from vinylitics.preproc.data import load_data

# if __name__ == '__main__':
#         df = load_data()

def recommend_track(track_name, artist, df:pd.DataFrame):
    track_name = track_name.lower().strip()
    artist = artist.lower().strip()



    selected_track = df[(df['track_name'] == track_name) & (df['artists'] == artist)]
    if selected_track.empty:
        # Deploy fuzzy search if track not found

        # Combine track name and artist for fuzzy matching
        query = track_name + " " + artist
        # Create a list of combined strings for each track in the dataframe
        choices = (df['track_name'].str.lower() + " " + df['artists'].str.lower()).tolist()
        # Get the top 3 matches
        best_matches = process.extract(query, choices, limit=3, scorer=fuzz.token_set_ratio)
        print(f"No exact match found for '{track_name} {artist}'. Did you mean:")
        for i, match in enumerate(best_matches, 1):
            print(f"{i}. {match[0]} (Score: {match[1]})")
        try:
            selection = int(input("Enter the number of the correct match (or 0 to cancel): "))
        except ValueError:
            print("Invalid input. Cancelling selection.")
            return None
        if selection == 0 or selection > len(best_matches):
            print("No track selected.")
            return None
        else:
            # best_matches returns tuples: (match_string, score)
            selected_match = best_matches[selection-1][0]
            # Find the index of selected_match in choices
            selected_index = choices.index(selected_match)
            selected_track = df.iloc[[selected_index]]

    # Continue with selection as before
    print("🎶 Track selected")
    track_preproc = preprocess_features(selected_track)

    distances, indices = find_neighbors(track_preproc)

    result = df.iloc[indices[0][1:]].sort_values(by='popularity', ascending=True)
    # print(result)
    if __name__ == '__main__':
        print(result)
    return result[['track_name', 'artists', 'track_id', 'popularity', 'tempo', 'danceability', 'energy', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence']], selected_track

if __name__ == "__main__":
    df = pd.read_csv("/Users/sbeasse/code/youssef-benk/vinylitics/raw_data/dataframe_2.csv")
    recommend_track("single ladies put a ring on it", "beyonce", df)
