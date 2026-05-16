import os
import numpy as np
import librosa
import soundfile as sf

def create_mock_dataset(target_dir="mock_dataset", num_files=60):
    """Generates synthetic .wav files with labels in the filename for testing."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Creating mock dataset at './{target_dir}'...")
        emotions = ["happy", "sad", "angry"]
        sr = 22050  # Sample rate
        duration = 2.0  # seconds
        
        for i in range(num_files):
            emotion = emotions[i % len(emotions)]
            # Generate random white noise as artificial speech signal
            dummy_signal = np.random.randn(int(sr * duration))
            filename = f"{i:03d}_{emotion}.wav"
            filepath = os.path.join(target_dir, filename)
            sf.write(filepath, dummy_signal, sr)
    else:
        print(f"Dataset directory './{target_dir}' already exists.")

def extract_mfcc(file_path, n_mfcc=40):
    """Loads an audio file and computes its mean MFCCs using modern Librosa API."""
    try:
        # Fixed: Removed resample_type argument to support Librosa 0.11.0+
        audio, sample_rate = librosa.load(file_path)
        
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        
        # Compress time axes to get a fixed-size vector for the model
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def load_and_prepare_data(dataset_path):
    """Loops through the folder to parse labels and extract audio features across platforms."""
    X, y = [], []
    
    # RAVDESS Emotion mapping
    emotion_map = {
        "01": "neutral", "02": "calm", "03": "happy", 
        "04": "sad", "05": "angry", "06": "fearful", 
        "07": "disgust", "08": "surprised"
    }
    
    # Platform-independent check for Windows backslashes and directory structures
    is_mock = os.path.basename(os.path.normpath(dataset_path)) == "mock_dataset"
    
    print(f"Extracting features from directory: {dataset_path}...")
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                
                if is_mock:
                    # Parse dummy filenames (e.g. '001_happy.wav')
                    try:
                        emotion = file.split("_")[1].replace(".wav", "")
                    except IndexError:
                        continue
                else:
                    # Parse real RAVDESS dataset hyphenated format
                    parts = file.split("-")
                    if len(parts) >= 3:
                        emotion_code = parts[2]
                        emotion = emotion_map.get(emotion_code)
                    else:
                        continue
                
                if emotion:
                    features = extract_mfcc(file_path)
                    if features is not None:
                        X.append(features)
                        y.append(emotion)
                            
    return np.array(X), np.array(y)