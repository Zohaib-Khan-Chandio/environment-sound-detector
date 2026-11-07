"""
selftest.py - check that the app pipeline reproduces your Colab results.

Usage (from the app folder):
    python selftest.py /path/to/100648-1-0-0.wav
    python selftest.py /path/to/100648-1-0-0.wav /path/to/urbansound8k_extracted_features.csv

Without a CSV it compares probabilities with the values printed in Part 4.
With a CSV it also compares the 47 features with that clip's row.
"""
import sys
from pathlib import Path

import numpy as np

from app_core import load_artifacts, prepare_audio, predict
from features import extract_features

# Printed in Part 4 for fold-10 clip 100648-1-0-0.wav (class order 0..9, rounded)
REFERENCE = np.array([4.00e-04, 3.30e-02, 9.00e-04, 8.70e-01, 8.15e-02,
                      1.20e-03, 8.80e-03, 9.00e-04, 2.20e-03, 1.10e-03])

wav = Path(sys.argv[1])
art = load_artifacts(Path(__file__).parent)

y, seconds = prepare_audio(wav.read_bytes())          # same route as an upload
res = predict(art, y)
print("clip length (s):", round(seconds, 2))
print("prediction:", res["label"], round(res["confidence"], 4))
print("probabilities:", np.round(res["proba"], 4))

# the in-memory route must equal the file-path route used in training
from features import features_from_array
same = np.allclose(features_from_array(y), extract_features(str(wav)), atol=1e-6)
print("bytes route == path route:", same)

if wav.name == "100648-1-0-0.wav":
    ok = np.allclose(res["proba"], REFERENCE, atol=0.006)
    print("matches Part 4 reference probabilities:", ok)

if len(sys.argv) > 2:
    import pandas as pd
    df = pd.read_csv(sys.argv[2])
    cols = art["feature_columns"]
    row = df.loc[df["slice_file_name"] == wav.name, cols].values[0]
    diff = np.abs(row - features_from_array(y)).max()
    print("max feature difference vs CSV:", diff, "->", "OK" if diff < 1e-3 else "MISMATCH")
