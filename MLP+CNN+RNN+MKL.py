import os
import pandas as pd
import numpy as np
import parselmouth
from parselmouth.praat import call
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.feature_selection import RFECV
from scikeras.wrappers import KerasClassifier  # Import KerasClassifier
import librosa

# Feature extraction function with additional features
def extract_voice_features(audio_file):
    sound = parselmouth.Sound(audio_file)

    # Extract pitch (fundamental frequency)
    pitch = call(sound, "To Pitch", 0.0, 75, 600)
    mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
    min_pitch = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    max_pitch = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")

    # Extract jitter (local)
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
    local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)

    # Extract shimmer (local)
    local_shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    # Extract harmonicity (HNR)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    mean_hnr = call(harmonicity, "Get mean", 0, 0)

    # Extract MFCCs and additional features
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mean_mfccs = np.mean(mfccs, axis=1)
    chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr), axis=1)
    rmse = np.mean(librosa.feature.rms(y=y))

    # Combine all features
    features = {
        "mean_pitch": mean_pitch,
        "min_pitch": min_pitch,
        "max_pitch": max_pitch,
        "local_jitter": local_jitter,
        "local_shimmer": local_shimmer,
        "mean_hnr": mean_hnr,
        "rmse": rmse,
        **{f'mfcc_{i+1}': mfcc for i, mfcc in enumerate(mean_mfccs)},
        **{f'chroma_{i+1}': chroma for i, chroma in enumerate(chroma_stft)},
        "spectral_rolloff": spectral_rolloff[0]
    }

    return features

def process_directory(directory):
    features_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                features = extract_voice_features(file_path)
                features['file'] = file  # Add filename for later reference
                features_list.append(features)
    return pd.DataFrame(features_list)

# Load and process the audio files
hc_directory = "C:/Users/mtshe/Downloads/HC_AH"
pd_directory = "C:/Users/mtshe/Downloads/PD_AH"
blind_directory = "C:/Users/mtshe/Downloads/Blind Data of 81"

hc_df = process_directory(hc_directory)
pd_df = process_directory(pd_directory)
blind_df = process_directory(blind_directory)

# Label the data
hc_df['label'] = 0  # Healthy Control
pd_df['label'] = 1  # Parkinson's Disease

# Combine the datasets
full_df = pd.concat([hc_df, pd_df], ignore_index=True)

# Split the data
X = full_df.drop(columns=['label', 'file']).values
y = full_df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_blind_scaled = scaler.transform(blind_df.drop(columns=['file']).values)

# Feature Selection using RFECV (Recursive Feature Elimination with Cross-Validation)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
selector = RFECV(rf, step=1, cv=5)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)
X_blind_selected = selector.transform(X_blind_scaled)

# Build the GRU model with selected features
def create_gru_model(input_shape):
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(input_shape, 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),
        Bidirectional(GRU(32, return_sequences=False)),  # Use GRU instead of LSTM
        Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),  # Added L1 regularization
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0003), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Wrap the Keras model in KerasClassifier
keras_gru_model = KerasClassifier(model=create_gru_model, input_shape=X_train_selected.shape[1], epochs=150, 
                                  batch_size=16, validation_split=0.2, 
                                  callbacks=[EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True), 
                                             ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=0.00001)],
                                  class_weight={0: 1.1, 1: 1.2})

# Random Forest model for ensemble
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_selected, y_train)

# Ensemble voting model (soft voting)
ensemble_model = VotingClassifier(estimators=[('gru', keras_gru_model), ('rf', rf_model)], voting='soft')
ensemble_model.fit(X_train_selected, y_train)  # No need to reshape for sklearn models

# Evaluate the ensemble model
ensemble_accuracy = ensemble_model.score(X_test_selected, y_test)
print(f"Ensemble Test Accuracy: {ensemble_accuracy * 100:.2f}%")

# Predict on the blind data using the ensemble model
blind_predictions = ensemble_model.predict(X_blind_selected)
blind_pred_labels = blind_predictions.astype(int).flatten()

# Compare predictions with actual labels (for test data)
comparison_df = pd.DataFrame({
    'file': blind_df['file'],
    'predicted': blind_pred_labels
})

# Print the prediction results
print(comparison_df)

# Save the prediction results
comparison_df.to_csv("blind_predictions.csv", index=False)

# Calculate prediction accuracy for HC and PD
hc_pred_correct = comparison_df[(comparison_df['predicted'] == 0) & (blind_df['file'].isin(hc_df['file']))].shape[0]
hc_pred_incorrect = comparison_df[(comparison_df['predicted'] == 1) & (blind_df['file'].isin(hc_df['file']))].shape[0]
pd_pred_correct = comparison_df[(comparison_df['predicted'] == 1) & (blind_df['file'].isin(pd_df['file']))].shape[0]
pd_pred_incorrect = comparison_df[(comparison_df['predicted'] == 0) & (blind_df['file'].isin(pd_df['file']))].shape[0]

# Detailed breakdown
print("\nTotal Predictions:")
print("1. Healthy Control (HC) Audio Files:")
print(f"   o Correctly Predicted as Healthy: {hc_pred_correct}")
print(f"   o Incorrectly Predicted as Parkinson's: {hc_pred_incorrect}")
print(f"   o Total HC Files: {hc_pred_correct + hc_pred_incorrect}")
print("2. Parkinson's Disease (PD) Audio Files:")
print(f"   o Correctly Predicted as Parkinson's: {pd_pred_correct}")
print(f"   o Incorrectly Predicted as Healthy: {pd_pred_incorrect}")
print(f"   o Total PD Files: {pd_pred_correct + pd_pred_incorrect}")