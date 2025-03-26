import os
import numpy as np
import pandas as pd
from scipy import stats
import librosa
import warnings
import dill
import tensorflow as tf

# Define absolute paths for the required files (assuming all files are in the same folder as this script)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "low_to_high_model.keras")
PCA_PATH = os.path.join(CURRENT_DIR, "pca_low_features.dill")
SCALER_LOW_PATH = os.path.join(CURRENT_DIR, "scaler_low_features.dill")
# SCALER_Y_PATH = os.path.join(CURRENT_DIR, "scaler_y.dill")
ORDERED_COLS_PATH = os.path.join(CURRENT_DIR, "low_level_features_ordered.csv")

# Expected moments for each feature
moments = ['mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max']

def get_ordered_columns():
    """
    Reads the ordered column names from low_level_features_ordered.csv.
    Assumes one column name per row.
    """
    try:
        ordered_cols = pd.read_csv(ORDERED_COLS_PATH, header=None)[0].tolist()
        # Remove first element if it is numeric (e.g. a row number)
        if ordered_cols and ordered_cols[0].strip().isdigit():
            ordered_cols = ordered_cols[1:]
        if __name__ == '__main__':
            print("Ordered columns loaded:", ordered_cols[:5], "...")
        return ordered_cols
    except Exception as e:
        if __name__ == '__main__':
            print("Error reading ordered columns:", e)
        raise

def feature_stats(name, values):
    """
    Given a feature (e.g. 'zcr') and its computed values (a 2D array),
    compute the seven statistics (as defined in moments) and store in a dict.
    """
    stat_funcs = {
        'mean': np.mean,
        'std': np.std,
        'skew': lambda v: stats.skew(v, axis=1),
        'kurtosis': lambda v: stats.kurtosis(v, axis=1),
        'median': np.median,
        'min': np.min,
        'max': np.max
    }
    feat_vals = {}
    try:
        v_min = np.nanmin(values)
        v_max = np.nanmax(values)
        v_mean = np.nanmean(values)
    except Exception as e:
        if __name__ == '__main__':
            print(f"Error computing summary for {name}: {e}")
        v_min, v_max, v_mean = None, None, None
    if __name__ == '__main__':
        print(f"Processing {name}: shape={values.shape}, min={v_min}, max={v_max}, mean={v_mean}")
    for stat in moments:
        func = stat_funcs[stat]
        try:
            # For skew and kurtosis, the function already computes over axis=1
            if stat in ['skew', 'kurtosis']:
                stat_vals = func(values)
            else:
                stat_vals = func(values, axis=1)
        except Exception as e:
            if __name__ == '__main__':
                print(f"Error computing {stat} for {name}: {e}")
            stat_vals = [0.0]
        for i, val in enumerate(stat_vals):
            key = f"{name}_{stat}_{i+1:02d}"
            if np.isnan(val):
                if __name__ == '__main__':
                    print(f"Warning: {key} is NaN, replacing with 0")
                val = 0.0
            feat_vals[key] = val
    return feat_vals

def extract_low_features_from_mp3(mp3_path: str) -> pd.DataFrame:
    """
    Extracts 518 low-level features from an mp3 file using the FMA methodology.
    Returns a DataFrame with one row and columns ordered as in low_level_features_ordered.csv.
    """
    feat_index = get_ordered_columns()
    feat_values = {}

    warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
    try:
        # Load the audio file (using mono)
        x, sr = librosa.load(mp3_path, sr=None, mono=True)
        if __name__ == '__main__':
            print(f"Loaded {mp3_path}: signal shape = {x.shape}, sr = {sr}")
    except Exception as e:
        if __name__ == '__main__':
            print(f"Error loading {mp3_path}: {repr(e)}")
        return pd.DataFrame()  # return empty if load fails

    try:
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
        feat_values.update(feature_stats('zcr', zcr))

        # Compute constant-Q transform (CQT) and derive chroma_cqt
        cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7*12))
        chroma_cqt = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        feat_values.update(feature_stats('chroma_cqt', chroma_cqt))

        # Chroma CENS from CQT
        chroma_cens = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        feat_values.update(feature_stats('chroma_cens', chroma_cens))

        # Tonnetz (using chroma_cens)
        tonnetz = librosa.feature.tonnetz(chroma=chroma_cens)
        feat_values.update(feature_stats('tonnetz', tonnetz))

        # STFT and Chroma STFT
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        chroma_stft = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
        feat_values.update(feature_stats('chroma_stft', chroma_stft))

        # RMS (Root Mean Square Energy)
        rms = librosa.feature.rms(S=stft)
        feat_values.update(feature_stats('rmse', rms))

        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(S=stft)
        feat_values.update(feature_stats('spectral_centroid', spectral_centroid))

        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=stft)
        feat_values.update(feature_stats('spectral_bandwidth', spectral_bandwidth))

        # Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(S=stft, n_bands=6)
        feat_values.update(feature_stats('spectral_contrast', spectral_contrast))

        # Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(S=stft)
        feat_values.update(feature_stats('spectral_rolloff', spectral_rolloff))

        # MFCCs
        mel = librosa.feature.melspectrogram(S=stft**2, sr=sr)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        feat_values.update(feature_stats('mfcc', mfcc))

    except Exception as e:
        if __name__ == '__main__':
            print(f"Error processing features for {mp3_path}: {repr(e)}")
        return pd.DataFrame()

    df_features = pd.DataFrame([feat_values])
    # Reindex to match the ordered columns; missing columns become NaN (fill later)
    df_features = df_features.reindex(columns=feat_index)
    if __name__ == '__main__':
        print("Extracted low-level features DataFrame:")
        print(df_features.head())
    df_features = df_features.fillna(0)
    return df_features

def predict_high_features(mp3_path: str) -> pd.DataFrame:
    """
    Uses the extracted low-level features, preprocessor objects, and the saved TensorFlow model
    to predict high-level features from an mp3 file.
    Also computes the tempo directly from the audio.
    Returns a final DataFrame with high-level features in the order:
      'tempo', 'danceability', 'energy', 'speechiness', 'acousticness',
      'instrumentalness', 'liveness', 'valence'
    """
    df_low = extract_low_features_from_mp3(mp3_path)
    if df_low.empty:
        print("No low-level features extracted. Aborting prediction.")
        return pd.DataFrame()

    if __name__ == '__main__':
        print("Low-level features before scaling:")
        print(df_low.describe().transpose().head())

    # Load the preprocessor objects and the model
    with open(SCALER_LOW_PATH, "rb") as f:
        scaler_low = dill.load(f)
    with open(PCA_PATH, "rb") as f:
        pca_low = dill.load(f)
    model = tf.keras.models.load_model(MODEL_PATH)

    try:
        X_scaled = scaler_low.transform(df_low)
        if __name__ == '__main__':
            print("Scaled low-level features shape:", X_scaled.shape)
            print("Scaled low-level features (first row):", X_scaled[0, :5])
    except Exception as e:
        if __name__ == '__main__':
            print("Error during scaling low-level features:", e)
        return pd.DataFrame()

    try:
        X_pca = pca_low.transform(X_scaled)
        if __name__ == '__main__':
            print("PCA-reduced features shape:", X_pca.shape)
            print("PCA output (first row):", X_pca[0, :5])
    except Exception as e:
        if __name__ == '__main__':
            print("Error during PCA transformation:", e)
        return pd.DataFrame()

    # After obtaining PCA-reduced features (X_pca) and predicting scaled y values:
    y_pred_scaled = model.predict(X_pca)
    if __name__ == '__main__':
        print("Scaled predictions:", y_pred_scaled)

    # Partition the scaled predictions according to the training order of high-level features.
    # The model was trained to predict 7 features in this order:
    # [Acousticness, Danceability, Energy, Instrumentalness, Liveness, Speechiness, Valence]
    # We now want to force instrumentalness to 0, so we will override that later.
    y_ushaped_scaled = y_pred_scaled[:, [0, 3]]
    y_even_scaled = y_pred_scaled[:, [1, 2, 6]]
    y_skewed_scaled = y_pred_scaled[:, [4, 5]]

    # Load the separate scaler objects for y:
    with open(os.path.join(CURRENT_DIR, 'scaler_y_even.dill'), 'rb') as f:
        scaler_y_even = dill.load(f)
    with open(os.path.join(CURRENT_DIR, 'scaler_y_ushaped.dill'), 'rb') as f:
        scaler_y_ushaped = dill.load(f)
    with open(os.path.join(CURRENT_DIR, 'scaler_y_skewed.dill'), 'rb') as f:
        scaler_y_skewed = dill.load(f)

    # Inverse-transform each group separately
    y_ushaped_inv = scaler_y_ushaped.inverse_transform(y_ushaped_scaled)
    y_even_inv = scaler_y_even.inverse_transform(y_even_scaled)
    y_skewed_inv = scaler_y_skewed.inverse_transform(y_skewed_scaled)

    # Create DataFrames for each group with the original column names used during training.
    # For u_shaped, the original order was:
    # ['echonest_audio_features_instrumentalness', 'echonest_audio_features_acousticness']
    df_ushaped = pd.DataFrame(y_ushaped_inv,
                              columns=['echonest_audio_features_instrumentalness', 'echonest_audio_features_acousticness'],
                              index=df_low.index)
    # We want acousticness first and then instrumentalness, but force instrumentalness to 0.
    df_ushaped = df_ushaped[['echonest_audio_features_acousticness', 'echonest_audio_features_instrumentalness']]
    df_ushaped["echonest_audio_features_instrumentalness"] = 0.0

    # For even, the original order was:
    # ['echonest_audio_features_energy', 'echonest_audio_features_danceability', 'echonest_audio_features_valence']
    df_even = pd.DataFrame(y_even_inv,
                           columns=['echonest_audio_features_energy', 'echonest_audio_features_danceability', 'echonest_audio_features_valence'],
                           index=df_low.index)
    # Reorder them to: danceability, energy, valence.
    df_even = df_even[['echonest_audio_features_danceability', 'echonest_audio_features_energy', 'echonest_audio_features_valence']]

    # For skewed, the order is:
    # ['echonest_audio_features_liveness', 'echonest_audio_features_speechiness']
    df_skewed = pd.DataFrame(y_skewed_inv,
                             columns=['echonest_audio_features_liveness', 'echonest_audio_features_speechiness'],
                             index=df_low.index)

    # Define the desired final column order (7 features, excluding tempo which is computed separately)
    final_order = [
        'echonest_audio_features_acousticness',    # from u_shaped
        'echonest_audio_features_danceability',      # from even
        'echonest_audio_features_energy',            # from even
        'echonest_audio_features_instrumentalness',    # from u_shaped (forced to 0)
        'echonest_audio_features_liveness',          # from skewed
        'echonest_audio_features_speechiness',       # from skewed
        'echonest_audio_features_valence'            # from even
    ]

    # Merge the inverse-transformed DataFrames
    df_pred = pd.concat([df_ushaped, df_even, df_skewed], axis=1)
    # Reorder columns to match the final desired order
    df_pred = df_pred[final_order]

    if __name__ == '__main__':
        print("Inverse-transformed predictions (after forcing instrumentalness to 0):")
        print(df_pred)

    # Compute tempo directly from the audio using librosa.beat.beat_track
    try:
        y_audio, sr = librosa.load(mp3_path, sr=None, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y_audio, sr=sr)
        tempo = tempo * 2  # half-time adjustment
        if __name__ == '__main__':
            print(f"Computed tempo from audio: {tempo}")
    except Exception as e:
        if __name__ == '__main__':
            print("Error computing tempo from audio:", e)
        tempo = 0.0

    # Reassemble final DataFrame with correct column order for nearest neighbors search:
    # Order: 'tempo', 'danceability', 'energy', 'speechiness',
    #        'acousticness', 'instrumentalness', 'liveness', 'valence'
    final_data = {
        "tempo": float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo),
        "danceability": float(df_pred["echonest_audio_features_danceability"].iloc[0]),
        "energy": float(df_pred["echonest_audio_features_energy"].iloc[0]),
        "speechiness": float(df_pred["echonest_audio_features_speechiness"].iloc[0]),
        "acousticness": float(df_pred["echonest_audio_features_acousticness"].iloc[0]),
        "instrumentalness": float(df_pred["echonest_audio_features_instrumentalness"].iloc[0]),
        "liveness": float(df_pred["echonest_audio_features_liveness"].iloc[0]),
        "valence": float(df_pred["echonest_audio_features_valence"].iloc[0])
    }

    df_high = pd.DataFrame([final_data])
    return df_high

# For testing:
if __name__ == '__main__':
    test_mp3_path = "/Users/adviti/code/youssef-benk/vinylitics/The Weeknd - Blinding Lights (Official Audio).mp3"
    print("Extracting high-level features for:", test_mp3_path)
    df_high = predict_high_features(test_mp3_path)
    print("Predicted high-level features:")
    print(df_high)
