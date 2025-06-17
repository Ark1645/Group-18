import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model("skin_cancer_model.h5")

# Define class labels
class_labels = ['Benign', 'Malignant']  # or your actual class names

st.title("Skin Cancer Detection")
st.write("Upload a skin lesion image and the model will predict if it's benign or malignant.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = class_labels[int(prediction > 0.5)]
    confidence = prediction if label == "Malignant" else 1 - prediction

    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"### Confidence: **{confidence * 100:.2f}%**")
