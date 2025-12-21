
'''import streamlit as st
import cv2
import numpy as np
import pickle
import os
import tempfile
from tensorflow.keras.models import load_model
from googletrans import Translator
from PIL import Image
import io
from gtts import gTTS
import pygame

# Load pre-trained models and label encoder
model_vgg16 = load_model("vgg16.keras")
model_vgg19 = load_model("vgg19.keras")

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Initialize translator
translator = Translator()

# Function to preprocess images for model prediction
def preprocess_image(image_path, target_size=(124, 124)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Function to predict the class from the image using the model
def predict_class(image, model):
    if isinstance(image, str):  # If image is a file path
        image = preprocess_image(image)
    else:  # If image is in-memory
        image = np.array(Image.open(io.BytesIO(image)))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (124, 124)) / 255.0
        image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class_index = np.argmax(prediction)
    predicted_class = label_encoder.inverse_transform([predicted_class_index])
    return predicted_class[0]

# Convert class name to Urdu and then to English
def translate_class_to_urdu_and_english(class_name):
    try:
        urdu_translation = translator.translate(class_name, src='en', dest='ur').text
        english_translation = translator.translate(urdu_translation, src='ur', dest='en').text
        return urdu_translation, english_translation
    except Exception as e:
        return f"Translation Error: {str(e)}", f"Translation Error: {str(e)}"

# Function to generate and play audio from text
def play_audio(text):
    try:
        tts = gTTS(text=text, lang='ur')  # Generate speech
        
        # Save temporary file
        temp_audio_path = "temp_audio.mp3"
        tts.save(temp_audio_path)

        # Initialize Pygame mixer
        pygame.mixer.init()
        pygame.mixer.music.load(temp_audio_path)
        pygame.mixer.music.play()

        # Wait until audio is finished playing
        while pygame.mixer.music.get_busy():
            pygame.time.delay(100)

        # Clean up
        pygame.mixer.quit()
        os.remove(temp_audio_path)
    except Exception as e:
        st.error(f"Audio playback failed: {str(e)}")

# Login Page (to be shown initially)
def login_page():
    st.title("Login")
    username = st.text_input("Username", "")
    password = st.text_input("Password", "", type="password")
    
    if st.button("Login"):
        if username == "Admin" and password == "123":
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.experimental_rerun()
  # Refresh the page to show the app content after login
        else:
            st.error("Invalid username or password!")

# Displaying content only after login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# If user is not logged in, only show the login page
if not st.session_state.logged_in:
    login_page()
else:
    # Show sidebar and app content after login
    st.sidebar.title("Indo-Pak Sign Language Prediction")
    selection = st.sidebar.radio("Choose a Page", ["Home", "Image Prediction", "Camera Prediction"])

    # Home Page
    if selection == "Home":
        st.title("Welcome to Indo-Pak Sign Language Prediction")
        st.write("""
            **Project Name**: Indo-Pak Sign Language Prediction
            This project aims to predict Indo-Pak sign language gestures using deep learning models. 
            The system is based on a dataset of hand gestures representing letters of the Indo-Pak Sign Language. 
            It uses a Convolutional Neural Network (CNN) model (VGG16 or VGG19) to classify the gestures. The results 
            are displayed in both Urdu and English for ease of understanding.
        """)

    # Image Prediction Page
    elif selection == "Image Prediction":
        st.title("Sign Language Prediction from Image")
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

            # Model Selection (VGG16 or VGG19)
            model_choice = st.radio("Select Model", ("VGG16", "VGG19"))
            
            # Prediction Button
            if st.button("Predict"):
                if model_choice == "VGG16":
                    model = model_vgg16
                else:
                    model = model_vgg19

                # Save uploaded image to a temporary file
                image_bytes = uploaded_image.read()
                img = Image.open(io.BytesIO(image_bytes))
                img_path = "temp_image.jpg"
                img.save(img_path)

                predicted_class = predict_class(img_path, model)
                urdu_class, english_class = translate_class_to_urdu_and_english(predicted_class)

                st.write(f"Predicted: {urdu_class}")
                st.write(f"Translated to English: {english_class}")
                play_audio(urdu_class)
                play_audio(english_class)
                # Cleanup temp file
                if os.path.exists(img_path):
                    os.remove(img_path)

    # Camera Prediction Page
    elif selection == "Camera Prediction":
        st.title("Sign Language Prediction from Camera")
        
        # Use Streamlit's camera input for capturing the image
        captured_image = st.camera_input("Capture a photo for prediction")
        
        if captured_image is not None:
            # Display the captured image
            st.image(captured_image, caption="Captured Image", use_column_width=True)
            
            # Save the captured image to a temporary file
            img_path = "temp_camera_image.jpg"
            with open(img_path, "wb") as f:
                f.write(captured_image.getvalue())
            
            # Prediction
            model_choice = st.radio("Select Model", ("VGG16", "VGG19"))
            if model_choice == "VGG16":
                model = model_vgg16
            else:
                model = model_vgg19
            
            predicted_class = predict_class(img_path, model)
            urdu_class, english_class = translate_class_to_urdu_and_english(predicted_class)

            st.write(f"Translated to Urdu: {urdu_class}")
            st.write(f"Translated back to English: {english_class}")

            # Play audio for predicted classes
            play_audio(urdu_class)
            play_audio(english_class)

            # Cleanup temp file
            if os.path.exists(img_path):
                os.remove(img_path)
'''

import streamlit as st
import cv2
import numpy as np
import pickle
import io
from PIL import Image
from tensorflow.keras.models import load_model
from googletrans import Translator
from gtts import gTTS

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Indo-Pak Sign Language",
    layout="centered"
)

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    vgg16 = load_model("vgg16.keras")
    vgg19 = load_model("vgg19.keras")
    return vgg16, vgg19

@st.cache_resource
def load_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)

model_vgg16, model_vgg19 = load_models()
label_encoder = load_encoder()

translator = Translator()

# -------------------- IMAGE PREPROCESS --------------------
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (124, 124))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------- PREDICTION --------------------
def predict(img, model):
    pred = model.predict(img)
    idx = np.argmax(pred)
    return label_encoder.inverse_transform([idx])[0]

# -------------------- AUDIO --------------------
def play_audio(text, lang):
    tts = gTTS(text=text, lang=lang)
    audio = io.BytesIO()
    tts.write_to_fp(audio)
    st.audio(audio.getvalue(), format="audio/mp3")

# -------------------- LOGIN --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.title("🔐 Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "Admin" and pwd == "123":
            st.session_state.logged_in = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

if not st.session_state.logged_in:
    login_page()
    st.stop()

# -------------------- MAIN APP --------------------
st.sidebar.title("Indo-Pak Sign Language")
page = st.sidebar.radio("Navigate", ["Home", "Image Prediction", "Camera Prediction"])

# -------------------- HOME --------------------
if page == "Home":
    st.title("🤟 Indo-Pak Sign Language Recognition")
    st.write("""
    This application recognizes **Indo-Pak sign language gest**

