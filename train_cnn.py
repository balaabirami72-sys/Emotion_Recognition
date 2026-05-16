import os
# Suppress initial TensorFlow system configuration logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Import the utility functions we wrote in the other file
from data_utils import load_and_prepare_data

if __name__ == "__main__":
    # 1. Point to your real folder containing the RAVDESS .wav files
    DATASET_DIR = "ravdess_data"
    
    # 2. Load audio features and emotion targets
    X, y = load_and_prepare_data(DATASET_DIR)
    if len(X) == 0:
        print(f"Error: No data found in './{DATASET_DIR}'. Please verify your dataset path.")
        exit()
        
    print(f"\nData Loaded. Features shape: {X.shape}, Labels shape: {y.shape}")
    
    # 3. Encode categorical text labels to One-Hot binary matrices
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(np.unique(y_encoded))
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)
    
    # Save the label classes tracking array so our predictor script can decode outputs later
    np.save('classes.npy', label_encoder.classes_)
    
    # 4. Split data into training (80%) and validation sets (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
    
    # Reshape features to fit 1D CNN expected layout: (samples, steps, channels)
    X_train = np.expand_dims(X_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    
    # 5. Build 1D Convolutional Neural Network
    model = Sequential([
        Conv1D(64, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Conv1D(128, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # Set up checkpoint to save the best configuration based on validation accuracy
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    
    # 6. Train the CNN model
    print("\nTraining the Speech Emotion Recognition Model... (Please hold, initialization might take a minute)")
    model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test), callbacks=[checkpoint])
    
    # 7. Evaluate on held-out test split
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[Test Result] Final Loss: {loss:.4f} | Final Accuracy: {accuracy * 100:.2f}%")
    print("Model file 'best_model.h5' and classes tracking array 'classes.npy' saved successfully.")