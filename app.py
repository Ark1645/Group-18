import os
import requests
import streamlit as st
import numpy as np
from PIL import Image
import os
import requests
from tensorflow.keras.models import load_model

# Download model from Google Drive
MODEL_URL = "https://drive.google.com/file/d/1qrCihaAdrvnNSsOUshchJxasA9ZvYwem/view?usp=sharing"
MODEL_PATH = "skin_cancer_model.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with requests.get(MODEL_URL, stream=True) as r:
            with open(MODEL_PATH, 'wb') as f:
                f.write(r.content)

download_model()
model = load_model(MODEL_PATH)