import streamlit as st

# Set page config FIRST!
st.set_page_config(
    page_title="Indo-Pak Sign Language Real-Time Detection",
    layout="wide"
)

import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from googletrans import Translator
from PIL import Image
import io
from gtts import gTTS
import tempfile
import os
import pygame

# Load models and label encoder
@st.cache_resource
def load_models():
    model_vgg16 = load_model("vgg16.keras")
    model_vgg19 = load_model("vgg19.keras")
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model_vgg16, model_vgg19, label_encoder

model_vgg16, model_vgg19, label_encoder = load_models()

translator = Translator()

def preprocess_frame(frame, target_size=(124, 124)):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def predict_class(frame, model):
    image = preprocess_frame(frame)
    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = label_encoder.inverse_transform([predicted_class_index])
    return predicted_class[0]

def translate_class_to_urdu_and_english(class_name):
    try:
        urdu_translation = translator.translate(class_name, src='en', dest='ur').text
        english_translation = translator.translate(urdu_translation, src='ur', dest='en').text
        return urdu_translation, english_translation
    except Exception as e:
        return f"Translation Error: {str(e)}", f"Translation Error: {str(e)}"

def play_audio(text, lang='ur'):
    try:
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            pygame.mixer.init()
            pygame.mixer.music.load(fp.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.delay(100)
            pygame.mixer.quit()
            os.remove(fp.name)
    except Exception as e:
        st.error(f"Audio playback failed: {str(e)}")

def play_prediction_audio(urdu_text, english_text):
    play_audio(urdu_text, lang='ur')
    play_audio(english_text, lang='en')

# --- Streamlit App ---
# Login Page
def login_page():
    st.title("Login")
    username = st.text_input("Username", "")
    password = st.text_input("Password", "", type="password")
    if st.button("Login"):
        if username == "Admin" and password == "123":
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password!")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_page()
    st.stop()

# Sidebar Navigation
st.sidebar.title("Indo-Pak Sign Language Prediction")
selection = st.sidebar.radio("Choose a Page", ["Home", "Image Prediction", "Real-Time Detection"])

if selection == "Home":
    st.title("Welcome to Indo-Pak Sign Language Prediction")
    st.write("""
        **Project Name:** Indo-Pak Sign Language Prediction  
        This project predicts Indo-Pak sign language gestures using deep learning (VGG16/VGG19).  
        Results are displayed in Urdu and English, with audio feedback.
    """)

elif selection == "Image Prediction":
    st.title("Sign Language Prediction from Image")
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    model_choice = st.radio("Select Model", ("VGG16", "VGG19"))
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        if st.button("Predict"):
            model = model_vgg16 if model_choice == "VGG16" else model_vgg19
            image = Image.open(uploaded_image)
            img_array = np.array(image)
            predicted_class = predict_class(img_array, model)
            urdu_class, english_class = translate_class_to_urdu_and_english(predicted_class)
            st.write(f"Predicted (Urdu): {urdu_class}")
            st.write(f"Predicted (English): {english_class}")
            play_prediction_audio(urdu_class, english_class)

elif selection == "Real-Time Detection":
    st.title("Real-Time Sign Language Detection (Webcam)")
    model_choice = st.radio("Select Model", ("VGG16", "VGG19"))
    run_detection = st.checkbox("Start Webcam Detection")
    FRAME_WINDOW = st.empty()

    if run_detection:
        model = model_vgg16 if model_choice == "VGG16" else model_vgg19
        cap = cv2.VideoCapture(0)
        st.info("Press 'Stop' to end detection.")

        prev_prediction = None
        try:
            while run_detection:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam.")
                    break
                predicted_class = predict_class(frame, model)
                urdu_class, english_class = translate_class_to_urdu_and_english(predicted_class)
                # Overlay prediction
                cv2.putText(frame, f"Urdu: {urdu_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(frame, f"Eng: {english_class}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # Only play audio if prediction changes (to avoid repeating the same audio rapidly)
                if prev_prediction != (urdu_class, english_class):
                    play_prediction_audio(urdu_class, english_class)
                    prev_prediction = (urdu_class, english_class)
        finally:
            cap.release()
