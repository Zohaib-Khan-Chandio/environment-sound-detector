"""
app_core.py
Loading, preparing and predicting. No Streamlit code here, so it can be
tested in Colab or a terminal (see selftest.py).
"""
import json
from pathlib import Path

import joblib
import numpy as np

from features import (TARGET_SR, TARGET_LEN, N_FEATURES,
                      load_audio, fix_length, features_from_array)

SILENCE_PEAK = 1e-4   # below this peak amplitude the clip is treated as silent


def load_artifacts(folder):
    """Load model + metadata and check they agree with each other."""
    folder = Path(folder)
    model = joblib.load(folder / "final_model.pkl")
    class_names = json.load(open(folder / "class_names.json"))
    feature_columns = json.load(open(folder / "feature_columns.json"))
    config = json.load(open(folder / "config.json"))

    expected = [f"feat_{i}" for i in range(1, N_FEATURES + 1)]
    if feature_columns != expected:
        raise ValueError("feature_columns.json does not match feat_1 ... feat_47.")
    n_in = getattr(model, "n_features_in_", N_FEATURES)
    if n_in != N_FEATURES:
        raise ValueError(f"Model expects {n_in} features, app produces {N_FEATURES}.")
    if len(class_names) != len(model.classes_):
        raise ValueError("class_names.json and the model's classes do not match.")
    return {"model": model, "class_names": class_names,
            "feature_columns": feature_columns, "config": config}


def prepare_audio(source):
    """Load a clip and fix it to 4 seconds.
    Returns (y_4s, original_seconds). Raises ValueError with a readable message."""
    try:
        y_full = load_audio(source)
    except Exception as exc:
        raise ValueError("Could not read this file. Please upload a valid WAV file.") from exc
    if y_full.size == 0:
        raise ValueError("This file contains no audio.")
    return fix_length(y_full), len(y_full) / TARGET_SR


def is_silent(y):
    return float(np.max(np.abs(y))) < SILENCE_PEAK


def predict(artifacts, y):
    """Run the model on a 4-second clip.
    The model is a Pipeline (scaler inside), so features go in unscaled."""
    model = artifacts["model"]
    names = artifacts["class_names"]
    x = features_from_array(y).reshape(1, -1)
    proba = model.predict_proba(x)[0]
    order = np.argsort(proba)[::-1]
    ranked = [(names[int(model.classes_[i])], float(proba[i])) for i in order]
    return {"label": ranked[0][0], "confidence": ranked[0][1],
            "top3": ranked[:3], "proba": proba}
