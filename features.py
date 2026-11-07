"""
features.py
Feature extraction for the Environmental Sound Detector.

This is the SAME logic as the final extract_features from Part 2 (the version
that built urbansound8k_extracted_features.csv), split into small pieces so the
app can also use audio that is already in memory (uploads, microphone).

Do not change any setting here without retraining the model.
"""
import io
import warnings

import numpy as np
import librosa

TARGET_SR = 22050
DURATION = 4.0
N_MFCC = 13
TARGET_LEN = int(TARGET_SR * DURATION)   # 88,200 samples
N_FEATURES = 47


def load_audio(source):
    """Load a path, file-like object or raw bytes as mono float32 at 22050 Hz.
    The clip keeps its full length. Raises an exception if it cannot be read."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    y, _ = librosa.load(source, sr=TARGET_SR, mono=True)
    return y


def fix_length(y):
    """Pad with zeros or trim to exactly 4 seconds (same as training)."""
    if len(y) < TARGET_LEN:
        return np.pad(y, (0, TARGET_LEN - len(y)), mode="constant")
    return y[:TARGET_LEN]


def features_from_array(y, sr=TARGET_SR):
    """Return the 47 features for a 4-second clip already loaded at 22050 Hz."""
    features = []

    # 1. MFCCs (26)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # 2. Delta MFCCs (2)
    delta = librosa.feature.delta(mfcc)
    features.append(np.mean(delta))
    features.append(np.std(delta))

    # 3. Delta-2 MFCCs (2)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features.append(np.mean(delta2))
    features.append(np.std(delta2))

    # 4. Spectral centroid (2)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(spec_cent))
    features.append(np.std(spec_cent))

    # 5. Spectral rolloff (2)
    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(spec_roll))
    features.append(np.std(spec_roll))

    # 6. Spectral bandwidth (2)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(spec_bw))
    features.append(np.std(spec_bw))

    # 7. Zero crossing rate (2)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # 8. RMS energy (2)
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.std(rms))

    # 9. Chroma STFT (7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=7, tuning=0)
    features.extend(np.mean(chroma, axis=1))

    features = np.array(features, dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def extract_features(file_path):
    """Same behaviour as the Part 2 function: path in, 47 features out
    (or None if the file cannot be read)."""
    try:
        y = fix_length(load_audio(file_path))
        return features_from_array(y)
    except Exception:
        return None
