# 🤟 Indo-Pak Sign Language Recognition System

**Real-Time Sign Language Detection using Deep Learning**

## 📌 Project Overview

The **Indo-Pak Sign Language Recognition System** is an end-to-end deep learning application that detects **Indo-Pak sign language hand gestures** from images and live webcam input, translates them into **Urdu and English**, and provides **audio output**.

The system is built using **transfer learning (VGG16 & VGG19)** and deployed as an interactive **Streamlit web application** with authentication and real-time prediction support.

---

## ✨ Features

* 🔐 Login-protected Streamlit application
* 📷 Image-based sign language prediction
* 🎥 Real-time webcam sign detection
* 🧠 Model selection: **VGG16 / VGG19**
* 🌍 Automatic translation (English ↔ Urdu)
* 🔊 Text-to-Speech output
* 📊 High-accuracy CNN models
* ⚡ Works on both **CPU and GPU**

---

## 🧠 Tech Stack

* **Programming Language**: Python
* **Deep Learning**: TensorFlow, Keras
* **Models**: VGG16, VGG19 (Transfer Learning)
* **Web Framework**: Streamlit
* **Image Processing**: OpenCV, PIL
* **NLP**: GoogleTrans
* **Audio**: gTTS, Pygame
* **Data Handling**: NumPy, Pandas
* **Visualization**: Matplotlib, Seaborn

---

## 📂 Dataset Details

* 📦 Total Images: **~34,000+**
* 🏷️ Classes: **37 Indo-Pak sign characters**
* 🖼️ Image Size: **128×128** (resized to 124×124)
* 🔁 Balanced using resampling
* 🎨 Augmented dataset for robustness

---

## 📈 Model Performance

### 🔹 VGG19

* Validation Accuracy: **~99%**
* Strong generalization
* Stable convergence

### 🔹 VGG16

* Validation Accuracy: **~99.5%**
* Faster training
* Lower computational cost

Evaluation metrics used:

* Confusion Matrix
* Precision, Recall, F1-Score
* Classification Report

---

## 🧪 What I Learned

* Transfer learning with pre-trained CNNs
* Handling large image datasets
* Class imbalance and resampling techniques
* CNN image preprocessing
* Model evaluation and tuning
* Real-time inference with webcam input
* ML deployment using Streamlit
* Integrating translation and speech synthesis
* Building a complete production-ready ML pipeline

---

## ⚠️ Challenges & How I Solved Them

| Challenge             | Solution                      |
| --------------------- | ----------------------------- |
| Class imbalance       | Resampling minority classes   |
| Overfitting           | Dropout + frozen base layers  |
| High GPU memory usage | Enabled GPU memory growth     |
| Slow inference        | Optimized image resolution    |
| Translation failures  | Added exception handling      |
| Audio crashes         | Controlled mixer lifecycle    |
| Webcam lag            | Streamlit camera optimization |

---

## ▶️ How to Run the Project

```bash
# Clone the repository
git clone https://github.com/your-username/indo-pak-sign-language-recognition.git
cd indo-pak-sign-language-recognition

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

> ⚠️ If you **do not have GPU/CUDA**, remove the GPU configuration cell — the project works fully on CPU.

---

🖥️ Application Screens (My Implementation)
![IMG_2887 2](https://github.com/user-attachments/assets/a7b10a18-296f-42bd-8e7a-0670bd252540)
![E4EEEC61-308D-4DE8-9617-752DCFA55EBF](https://github.com/user-attachments/assets/66e6c4b4-b684-4d1c-8eed-91ee51c9bce6)
![A92FA14A-CD79-47DF-A815-B6660F985D89](https://github.com/user-attachments/assets/55757e4c-4b13-459d-a6f6-f093bc9aaf49)

---

## 📌 Future Enhancements

* 🎯 Video-based gesture sequence recognition
* 📱 Mobile-friendly UI
* 🌍 Support for additional sign languages
* 🤖 Vision Transformers (ViT)
* ☁️ Cloud deployment (AWS / GCP)

---
## 🏗️ System Architecture

**Input**
- Uploaded image OR live webcam frame

**Preprocessing**
- Resize to 124×124
- Normalize pixel values

**Prediction**
- CNN model (VGG16 / VGG19)

**Post-Processing**
- Label decoding
- Urdu & English translation
- Audio generation

**Output**
- Text + Speech + UI visualization
  
---
## 👨‍💻 Author

This project demonstrates my hands-on experience in:

* Deep Learning & Computer Vision
* End-to-End ML system design
* Real-time inference systems
* Deployment-ready Python applications

---

## ⭐ Support

If you find this project useful:

* ⭐ Star the repository
* 🍴 Fork and experiment
* 🧠 Use it as a reference for ML + CV projects

---

