import pandas as pd
import os
import numpy as np
import parselmouth
from parselmouth.praat import call
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)  # Ensures no truncation for any column

def extract_voice_features(audio_file):
    try:
        sound = parselmouth.Sound(audio_file)
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
        
        y, sr = librosa.load(audio_file, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        return {
            "mean_pitch": mean_pitch,
            **{f'mfcc_{i+1}': mfcc for i, mfcc in enumerate(np.mean(mfccs, axis=1))}
        }
    except Exception as e:
        print(f"Failed to process {audio_file}: {e}")
        return None

def process_directory(directory):
    features_list = []
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return pd.DataFrame()

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                features = extract_voice_features(file_path)
                if features:
                    features['file'] = file_path
                    features_list.append(features)
                else:
                    print(f"Skipping file due to processing error: {file_path}")

    return pd.DataFrame(features_list)

hc_directory = r"C:\Users\mtshe\Downloads\HC_AH"
pd_directory = r"C:\Users\mtshe\Downloads\PD_AH"
blind_directory = r"C:\Users\mtshe\Downloads\Blind 81 Data"

hc_df = process_directory(hc_directory)
pd_df = process_directory(pd_directory)
blind_df = process_directory(blind_directory)

all_data_df = pd.concat([hc_df, pd_df, blind_df], ignore_index=True)

if all_data_df.empty:
    print("No data was processed. Please check the logs for errors.")
else:
    scaler = StandardScaler()
    X = scaler.fit_transform(all_data_df.drop(['file'], axis=1))
    y = np.concatenate([np.zeros(len(hc_df)), np.ones(len(pd_df)), -np.ones(len(blind_df))])

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X[:len(hc_df) + len(pd_df)], y[:len(hc_df) + len(pd_df)])

    probabilities = model.predict_proba(X)[:, 1]
    all_data_df['PD_Probability'] = probabilities
    all_data_df['Predicted_Label'] = (probabilities > 0.5).astype(int)

    print(all_data_df[['file', 'PD_Probability', 'Predicted_Label']])
    all_data_df.to_csv("full_predictions.csv", index=False)
