import sys
import os

# Add project root to sys.path so Python can find 'features' folder
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Now you can safely import your local modules
import sounddevice as sd
import soundfile as sf
from features.extract_features import extract_features
import joblib
import numpy as np
import time


# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Load trained binary model
model_path = os.path.join(project_root, "models/binary_model.pkl")
model = joblib.load(model_path)

def record_audio(duration=1, fs=44100, device=None):
    """Record audio from mic and return as numpy array"""
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, device=device)
    sd.wait()
    return audio.flatten(), fs

def monitor_mic(device=None):
    """Continuously monitor mic and predict Cry/No Cry every 1 second"""
    print("=== Live baby cry monitoring started (1 sec interval) ===")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            # Record 1 second chunk
            audio_data, fs = record_audio(duration=1, device=device)
            
            # Save temporarily for feature extraction
            temp_file = os.path.join(project_root, "temp_live.wav")
            sf.write(temp_file, audio_data, fs)
            
            # Extract features
            features = extract_features(temp_file)
            features = features.reshape(1, -1)
            
            # Predict
            prediction = model.predict(features)[0]
            print(f"[{time.strftime('%H:%M:%S')}] Prediction: {'Cry' if prediction == 1 else 'No Cry'}")

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

if __name__ == "__main__":
    # Optional: list available mics
    devices = sd.query_devices()
    input_devices = [idx for idx, d in enumerate(devices) if d['max_input_channels'] > 0]
    print("Available mics:")
    for idx in input_devices:
        print(f"{idx}: {devices[idx]['name']}")
    
    try:
        selected = int(input("Enter device ID (or press Enter for default): ") or -1)
    except ValueError:
        selected = -1
    if selected not in input_devices:
        selected = None  # use default mic

    monitor_mic(device=selected)
