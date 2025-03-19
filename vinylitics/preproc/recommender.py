

### **2️⃣ Finding Similar Songs** ###
def find_similar_songs(song_name, num_songs=5):
    """Searches for similar songs based on input song name."""
    matches = df[df["track_name"].str.contains(song_name, case=False, na=False)]

    if matches.empty:
        print("No matching song found. Try another name.")
        return None

    # Allow user to choose a song if multiple matches are found
    if len(matches) > 1:
        print("Multiple matches found. Choose the correct song:")
        for i, row in enumerate(matches.iterrows()):
            print(f"{i+1}: {row[1]['track_name']} by {row[1]['artists']}")
        choice = int(input("Enter the number of your choice: ")) - 1
        song_index = matches.index[choice]
    else:
        song_index = matches.index[0]

    print(f"✅ Found match: {df.loc[song_index, 'track_name']} by {df.loc[song_index, 'artists']}")

    # Get nearest neighbors
    distances, indices = knn.kneighbors([df_pca[song_index]], n_neighbors=num_songs+1)
    similar_songs = df.iloc[indices[0][1:]]  # Exclude the input song itself

    return similar_songs[['track_name', 'artists', 'track_genre', 'popularity', 'tempo']]
