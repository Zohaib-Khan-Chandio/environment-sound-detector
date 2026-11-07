# Environmental Sound Detector

Streamlit app for the UrbanSound8K project. Upload a clip (click or drag and drop)
or record with the microphone. The app shows the waveform, the Mel-spectrogram,
the predicted sound, its confidence and the top 3 classes. Below the threshold
(default 55%) it reports an unknown sound.

## Files
| File | Purpose |
|---|---|
| app.py | the web app |
| app_core.py | loading, padding/trimming, prediction |
| features.py | the 47-feature extraction (same logic as training) |
| selftest.py | checks the app gives the same probabilities as Colab |
| final_model.pkl, class_names.json, feature_columns.json, config.json | copy these from your Drive folder |
| requirements.txt, packages.txt | libraries |

## Run on your laptop
```
pip install -r requirements.txt
streamlit run app.py
```

## Run in Colab (demo)
Upload this folder to Drive and put the four model files in it. Then:
```
%cd /content/drive/MyDrive/EnvSoundDetector/EnvSoundApp
!pip install -q streamlit
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared
!nohup streamlit run app.py --server.port 8501 --server.headless true > streamlit.log 2>&1 &
!nohup ./cloudflared tunnel --url http://localhost:8501 > tunnel.log 2>&1 &
!sleep 10; grep -o 'https://[-a-z0-9]*\.trycloudflare\.com' tunnel.log | head -1
```
Open the printed https link. The microphone only works on https or localhost.

## Deploy (Streamlit Community Cloud)
Push the folder to GitHub, then create the app at share.streamlit.io with `app.py`.

## Check against Colab
```
python selftest.py path/to/100648-1-0-0.wav path/to/urbansound8k_extracted_features.csv
```
Dataset: UrbanSound8K (CC BY 4.0).
