![](./banner.png)
# vinylitics 🎵

> a gold mine for crate diggers.

tired of spotify recommending the same songs over and over? vinylitics is a music discovery app that takes a song you love and finds you **ten hidden gems**: tracks with a similar sound that never quite made it into the algorithm's spotlight.

built as the capstone project of the [le wagon](https://www.lewagon.com/) bootcamp, by a team of five over two weeks.

---

## what it does

you give vinylitics a song. it gives you ten you've probably never heard, but will want to.

under the hood, the app searches a database of **1 million songs** (sourced from a hugging face dataset scraped from the spotify api) and uses audio feature analysis and machine learning to find the closest sonic neighbours to your input, then deliberately surfaces only the least famous ones.

---

## how it works

**1. input**
the user enters a track name and artist.

**2. matching**
we look for the song in our database. if the spelling isn't exact, a fuzzy search handles common typos and variations before retrying the match.

**3. recommendation engine**
once we have a match, we use the song's spotify-engineered audio features (like danceability, acousticness, energy, etc.) and apply **PCA** + **nearest neighbour clustering** to find the most similar songs in the database.

**4. hidden gems filter**
from those neighbours, we return the **10 least famous** ones: the hidden gems.

---

## bonus feature (built, not shipped)

we also built an extended pipeline for songs that aren't in the database at all:

- the track is found on youtube and downloaded as an mp3
- a **neural network** processes the audio to reverse-engineer spotify-style features
- those features are fed into the same PCA + nearest neighbour pipeline to generate recommendations

this feature was fully developed but didn't make it into the final deployed app due to the coding deadline.

---

## tech stack

| layer | technology |
|---|---|
| api | fastapi, uvicorn |
| ml | scikit-learn, tensorflow, librosa |
| data | pandas, numpy, scipy |
| fuzzy search | thefuzz |
| audio | yt-dlp, ffmpeg |
| storage | google cloud storage, bigquery |
| containerisation | docker |

---

## sources
- [spotify tracks dataset, hugging face](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)
- [spotify 1.2M dataset, kaggle](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs)
- [spotify 1million track, kaggle](https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks)

## the team
⚡️ [dj flore](https://github.com/flore-perr)

🎸 [dj adviti](https://github.com/advitis)

💥 [mc stephanie](https://github.com/3Digitals-Agentur)

🎵 [mc youssef](https://github.com/youssef-benk)

📼 [dj suzie](https://github.com/SuzieBeasse)

## Let's dig in

→ frontend repo: [vinylitics_website](https://github.com/youssef-benk/vinylitics_website)

→ live app: [vinylitics.streamlit.app](https://vinylitics.streamlit.app/)
