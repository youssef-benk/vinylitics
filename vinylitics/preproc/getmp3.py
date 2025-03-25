import os

def get_mp3(song_name):
    """
    Downloads the first found YouTube result as an MP3 file
    https://pypi.org/project/yt-dlp/
    """
    import yt_dlp
    params = {
        'format': 'bestaudio/best',
        'outtmpl': '%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            # 'preferredquality': '192', # Uncomment to set quality by default it is best
        }],
    }
    with yt_dlp.YoutubeDL(params) as ydl:
        # Use extract_info to download and retrieve file metadata
        info = ydl.extract_info(f"ytsearch1:{song_name}", download=True)
        if 'entries' in info:
            info = info['entries'][0]
        filename = info.get('_filename')
        if not filename:
            title = info.get('title', 'downloaded')
            filename = f"{title}.mp3"
        # Convert filename to an absolute path
        abs_path = os.path.abspath(filename)
        print(f"Downloaded MP3: {abs_path}")
        return abs_path

# Deprecated function replaced by translator.py
# def get_mp3_features(song_name):
#     """
#     Downloads the MP3 for the given song using get_mp3 and processes it with librosa
#     Features extracted: MFCC, Spectral Centroid, Zero Crossing Rate, Chroma STFT, Tempo
#     https://librosa.org/doc/latest/index.html
#     """
#     import librosa
#     import pandas as pd
#     import numpy as np

#     # Download the MP3 file using the existing get_mp3 function
#     filename = get_mp3(song_name)

#     # Load the audio file using librosa
#     y, sr = librosa.load(filename, sr=None)  # Use the original sampling rate

#     # MFCC (Mel-Frequency Cepstral Coefficients) are a feature widely used in music and speech analysis
#     mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     mfcc_mean = np.mean(mfcc)

#     # Spectral Centroid is the center of mass of the spectrum
#     spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
#     spectral_centroid_mean = np.mean(spectral_centroid)

#     # Zero Crossing Rate is the rate of sign changes along a signal which indicates the presence of high-frequency content
#     zcr = librosa.feature.zero_crossing_rate(y)
#     zcr_mean = np.mean(zcr)

#     # STFT Chroma is a representation of the audio signal in the chroma domain which represents the 12 distinct pitch classes
#     chroma = librosa.feature.chroma_stft(y=y, sr=sr)
#     chroma_stft_mean = np.mean(chroma)

#     # Tempo denotes the speed or pace of a given piece and derives from the average time between beats
#     tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

#     # Putting it all together in a DataFrame
#     data = {
#         'mfcc_mean': [mfcc_mean],
#         'spectral_centroid_mean': [spectral_centroid_mean],
#         'zero_crossing_rate_mean': [zcr_mean],
#         'chroma_stft_mean': [chroma_stft_mean],
#         'tempo': [tempo]
#     }
#     df = pd.DataFrame(data)

#     print(df)
#     return df


# Test the function
import time
start_time = time.time()

get_mp3("Blinding lights The Weeknd (Official Audio)") # test to only get mp3
# get_mp3_features("Blinding lights The Weeknd") # test to get features

elapsed = time.time() - start_time
print(f"Time elapsed: {elapsed:.2f} seconds")
