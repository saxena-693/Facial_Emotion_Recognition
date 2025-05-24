import streamlit as st
import base64
import time
from tensorflow.keras.models import load_model



# Helper function to convert file to base64 string
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Set background image using base64 encoded PNG
def set_background(image_path):
    encoded = get_base64(image_path)
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Load custom pixel font from TTF file using base64
def load_custom_font(font_path):
    font_base64 = get_base64(font_path)
    font_css = f"""
    <style>
    @font-face {{
        font-family: 'PixelFont';
        src: url(data:font/ttf;base64,{font_base64}) format('truetype');
        font-weight: normal;
        font-style: normal;
    }}
    html, body, [class*="css"] {{
        font-family: 'PixelFont', monospace !important;
    }}
    </style>
    """
    st.markdown(font_css, unsafe_allow_html=True)

# === CALL THESE FUNCTIONS EARLY ===
set_background("background.png")
load_custom_font("pixel-font.ttf")

# === YOUR APP CONTENT STARTS HERE ===
st.markdown("""
    <div style='text-align: center; margin-top: 80px;'>
        <h1 style='font-size: 60px; font-family: PixelFont; color: #FFFFFF;'>Facial Emotion Recognition App</h1>
        <p style='font-size: 24px; font-family: PixelFont; color: #DDDDDD;'>Welcome to the App!!</p>
    </div>
""", unsafe_allow_html=True)

# Button + Progress Bar
if st.button("😈 Detect Emotion"):
    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.02)
        progress.progress(i + 1)
    st.success("🚀 Facial analysis complete!")


def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

model = load_model("emotion_model.h5")

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("emotion_model.h5")  # or "emotion_model.keras"

# Emotion labels in order (same as your model’s output)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emojis = ['😠', '🤢', '😱', '😄', '😢', '😲', '😐']

# Upload image
st.subheader("📤 Upload Your Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert to grayscale and resize to 48x48
    image = image.convert('L')  # L mode = grayscale
    image = image.resize((48, 48))

    # Convert to numpy array and normalize
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 48, 48)
    img_array = np.expand_dims(img_array, axis=-1)  # Shape: (1, 48, 48, 1)

    # Prediction button
    if st.button("😈 Detect Emotion", key="detect_emotion"):
        progress = st.progress(0)
        for i in range(100):
            import time

            time.sleep(0.01)
            progress.progress(i + 1)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)

        st.success("🚀 Facial analysis complete!")
        st.markdown(f"### Emotion: **{emotion_labels[predicted_class]}** {emojis[predicted_class]}")



