import os
import numpy as np
import librosa
import streamlit as st
from tensorflow.keras.models import load_model
from audio_recorder_streamlit import audio_recorder

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Speech Emotion Classifier", page_icon="🎙️", layout="centered")

st.title("🎙️ Speech Emotion Recognition System")
st.markdown("Analyze vocal features using a Deep Learning 1D-CNN pipeline to detect the underlying emotional state.")

# --- LOAD MODEL AND LABELS ---
@st.cache_resource
def load_ser_pipeline():
    model_path = 'best_model.h5' 
    classes_path = 'classes.npy'
    
    if not os.path.exists(model_path) or not os.path.exists(classes_path):
        return None, None
        
    model = load_model(model_path)
    classes = np.load(classes_path, allow_pickle=True)
    return model, classes

model, classes = load_ser_pipeline()

if model is None:
    st.error("⚠️ Error: 'best_model.h5' or 'classes.npy' not found! Please run your training script first.")
    st.stop()

# --- FEATURE EXTRACTION FUNCTION ---
def extract_features_from_buffer(audio_bytes):
    """Saves audio bytes temporarily to extract MFCC features using Librosa."""
    temp_filename = "temp_input_audio.wav"
    try:
        with open(temp_filename, "wb") as f:
            f.write(audio_bytes)
            
        # Load audio using the modern Librosa 0.11.0+ API
        audio, sample_rate = librosa.load(temp_filename)
        
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        # Extract 40 MFCCs matching your training setup
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        st.error(f"Error processing audio data: {e}")
        return None

# --- UI TABS: UPLOAD VS RECORD ---
tab1, tab2 = st.tabs(["📁 Upload Audio File", "🎙️ Record Live Audio"])
audio_data_bytes = None

with tab1:
    st.subheader("Upload an Audio Clip")
    uploaded_file = st.file_uploader("Choose a WAV audio file...", type=["wav"])
    if uploaded_file is not None:
        audio_data_bytes = uploaded_file.read()
        st.audio(audio_data_bytes, format="audio/wav")

with tab2:
    st.subheader("Record Voice via Microphone")
    st.write("Click the microphone icon below to start recording. Click it again to stop.")
    
    # Lightweight live recording widget
    recorded_audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#e74c3c",
        neutral_color="#95a5a6"
    )
    
    if recorded_audio_bytes is not None:
        audio_data_bytes = recorded_audio_bytes
        st.audio(audio_data_bytes, format="audio/wav")

# --- INFERENCE ENGINE ---
if audio_data_bytes is not None:
    st.markdown("---")
    if st.button("🚀 Analyze Speech Emotion", type="primary"):
        with st.spinner("Extracting signal features and executing neural layers..."):
            features = extract_features_from_buffer(audio_data_bytes)
            
            if features is not None:
                # Format to match model layout: (1 sample, 40 steps, 1 channel)
                features_reshaped = np.expand_dims(np.expand_dims(features, axis=0), axis=2)
                
                # Perform model prediction
                predictions = model.predict(features_reshaped)
                predicted_class_idx = np.argmax(predictions, axis=1)[0]
                predicted_emotion = classes[predicted_class_idx].upper()
                confidence = predictions[0][predicted_class_idx] * 100
                
                # --- DISPLAY RICH RESULTS ---
                st.subheader("🎯 Analysis Result")
                
                col1, col2 = st.columns(2)
                col1.metric(label="Predicted Emotion", value=predicted_emotion)
                col2.metric(label="Confidence Level", value=f"{confidence:.2f}%")
                
                st.markdown("##### Probability Breakdown:")
                for idx, emotion_label in enumerate(classes):
                    prob = predictions[0][idx]
                    st.write(f"**{emotion_label.capitalize()}**")
                    st.progress(float(prob))