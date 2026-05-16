import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from data_utils import extract_mfcc

if __name__ == "__main__":
    model_path = 'best_model.h5'
    classes_path = 'classes.npy'
    
    # Check if the model has been trained yet
    if not os.path.exists(model_path) or not os.path.exists(classes_path):
        print("Error: 'best_model.h5' or 'classes.npy' not found. Run 'train_cnn.py' first to train your model!")
        exit()
        
    # 1. Load trained assets
    model = tf.keras.models.load_model(model_path)
    classes = np.load(classes_path, allow_pickle=True)
    
    # 2. Define the path to the audio file you want to test
    # Change "mock_dataset/000_happy.wav" to your own custom audio path (e.g., "my_voice.wav") to test real predictions.
    test_audio_path = "mock_dataset/000_happy.wav" 
    
    if not os.path.exists(test_audio_path):
        print(f"File not found at: {test_audio_path}")
        exit()
        
    print(f"Analyzing audio file: {test_audio_path}...")
    
    # 3. Process file and pass through the input shape format layer
    features = extract_mfcc(test_audio_path)
    if features is not None:
        features = np.expand_dims(features, axis=0) # Add batch dimension
        features = np.expand_dims(features, axis=2) # Add channel dimension
        
        # 4. Predict
        predictions = model.predict(features, verbose=0)
        predicted_index = np.argmax(predictions)
        predicted_emotion = classes[predicted_index]
        confidence = predictions[0][predicted_index] * 100
        
        print("\n--- PREDICTION RESULT ---")
        print(f"Predicted Emotion: {predicted_emotion.upper()}")
        print(f"Confidence Level:  {confidence:.2f}%")
        print("-------------------------")