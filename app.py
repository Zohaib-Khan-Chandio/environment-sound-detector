"""
Environmental Sound Detector - Streamlit app (Part 5).
Run:  streamlit run app.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import streamlit as st

from features import TARGET_SR, DURATION
from app_core import load_artifacts, prepare_audio, is_silent, predict

APP_DIR = Path(__file__).parent
DEFAULT_THRESHOLD = 0.55

st.set_page_config(page_title="Environmental Sound Detector",
                   page_icon="🔊", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    return load_artifacts(APP_DIR)


try:
    art = get_artifacts()
except Exception as exc:
    st.error("The model files could not be loaded.")
    st.caption(f"Details: {exc}")
    st.stop()

CLASS_NAMES = art["class_names"]
pretty = lambda s: s.replace("_", " ")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Confidence threshold", 0.30, 0.90,
                          float(art["config"].get("threshold", DEFAULT_THRESHOLD)),
                          0.01,
                          help="Below this confidence the app reports an unknown sound.")
    st.caption("Project default: 55%.")
    st.subheader("Model")
    st.write("Tuned SVM (RBF) with a built-in scaler, trained on UrbanSound8K folds 1 to 9.")
    st.subheader("Sounds it knows")
    for name in CLASS_NAMES:
        st.write("• " + pretty(name))

# ---------------------------------------------------------------- header
st.title("🔊 Environmental Sound Detector")
st.write("Add a short sound clip and the app tells you which of 10 city sounds it hears, "
         "and how sure it is. If it is not sure, it says so.")


# ---------------------------------------------------------------- plots
def plot_waveform(y):
    fig, ax = plt.subplots(figsize=(6, 3))
    t = np.arange(len(y)) / TARGET_SR
    ax.plot(t, y, linewidth=0.6, color="#1f6f8b")
    ax.set_xlim(0, DURATION)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    fig.tight_layout()
    return fig


def plot_mel(y):
    S_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=128),
                               ref=np.max)
    fig, ax = plt.subplots(figsize=(6, 3))
    img = librosa.display.specshow(S_db, sr=TARGET_SR, x_axis="time", y_axis="mel",
                                   ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel-spectrogram")
    fig.tight_layout()
    return fig


def plot_top3(top3):
    fig, ax = plt.subplots(figsize=(6, 2.2))
    labels = [pretty(n) for n, _ in top3][::-1]
    values = [p * 100 for _, p in top3][::-1]
    ax.barh(labels, values, color="#1f6f8b")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)")
    for i, v in enumerate(values):
        ax.text(min(v + 1, 88), i, f"{v:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- analysis
def analyze(data, audio_format):
    if not data:
        st.error("This file is empty.")
        return
    st.audio(data, format=audio_format)

    with st.spinner("Analyzing..."):
        try:
            y, seconds = prepare_audio(data)
        except ValueError as exc:
            st.error(str(exc))
            return
        silent = is_silent(y)
        result = None if silent else predict(art, y)

    if seconds > DURATION + 0.05:
        st.info(f"This clip is {seconds:.1f} s long. Only the first 4 seconds were analyzed.")
    elif seconds < DURATION - 0.05:
        st.info(f"This clip is {seconds:.1f} s long. It was padded with silence to 4 seconds.")

    left, right = st.columns(2)
    for col, fig_fn in ((left, plot_waveform), (right, plot_mel)):
        with col:
            fig = fig_fn(y)
            st.pyplot(fig)
            plt.close(fig)

    if silent:
        st.warning("⚠️ Unknown / Uncertain Sound Detected. No sound was found in this clip.")
        return

    pct = result["confidence"] * 100
    if result["confidence"] >= threshold:
        st.success(f"### {pretty(result['label']).capitalize()}\n"
                   f"Confidence: **{pct:.1f}%**")
    else:
        st.warning(f"⚠️ Unknown / Uncertain Sound Detected (Low Confidence: {pct:.0f}%)")
        st.caption(f"Closest guess: {pretty(result['label'])}, "
                   f"below the {threshold * 100:.0f}% threshold.")

    st.subheader("Top 3 predictions")
    fig = plot_top3(result["top3"])
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Confidence is the probability the model gives to its top choice. "
               "A low value means the sound looks like several classes, or like none of them.")


# ---------------------------------------------------------------- inputs
tab_upload, tab_mic = st.tabs(["Upload a file", "Record from microphone"])

with tab_upload:
    st.write("Drag a file into the box, or click Browse files.")
    up = st.file_uploader("Sound file", type=["wav", "mp3", "flac", "ogg"],
                          label_visibility="collapsed")
    if up is not None:
        if not up.name.lower().endswith(".wav"):
            st.caption("WAV works best. If this file fails to load, convert it to WAV.")
        analyze(up.getvalue(), up.type or "audio/wav")
    else:
        st.caption("Waiting for a file. A clip of about 4 seconds works best.")

with tab_mic:
    if hasattr(st, "audio_input"):
        st.write("Record about 4 seconds with the sound source close to the microphone.")
        st.caption("The microphone only works on https:// links or on localhost. "
                   "Phone and laptop microphones add noise, so accuracy is lower than on dataset clips.")
        rec = st.audio_input("Record a sound")
        if rec is not None:
            analyze(rec.getvalue(), "audio/wav")
    else:
        st.info("This version of Streamlit has no recording widget. "
                "Upgrade Streamlit (1.39 or newer) or use the upload tab.")

# ---------------------------------------------------------------- footer
st.divider()
st.caption("Group name: ______  |  Dataset: UrbanSound8K (CC BY 4.0), "
           "J. Salamon, C. Jacoby and J. P. Bello, 2014.")
