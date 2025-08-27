"""
Baby Cry Detection & Reason Classification
------------------------------------------
- Stage 1: Binary (cry / no-cry) using MFCC + simple MLP
- Stage 2: Multi-class (hungry/diaper/discomfort/temperature/unknown) using log-mel spectrogram + CNN
- Saves .h5 and .tflite models
- Inference demo + notification stub

Install first:
  pip install numpy scipy librosa scikit-learn tensorflow==2.* soundfile

Run examples:
  # Train Stage 1 (binary)
  python baby_cry_ml.py train-binary --data ./dataset/binary --out ./models

  # Train Stage 2 (multi-class)
  python baby_cry_ml.py train-multiclass --data ./dataset/multiclass --out ./models

  # Infer with Stage 1
  python baby_cry_ml.py infer-binary --model ./models/binary_mlp.h5 --file ./test_audio/sample.wav

  # Infer with Stage 2
  python baby_cry_ml.py infer-multiclass --model ./models/mclass_cnn.h5 --file ./test_audio/sample.wav
"""

import os
import argparse
import glob
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras import layers, models

# -------------------------------
# Audio & feature utilities
# -------------------------------
TARGET_SR = 16000
CLIP_SECONDS = 3.0
CLIP_SAMPLES = int(TARGET_SR * CLIP_SECONDS)
N_MFCC = 20
N_MELS = 64
HOP_LENGTH = 256
N_FFT = 1024
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_clip(path, target_sr=TARGET_SR, clip_samples=CLIP_SAMPLES):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    if len(y) < clip_samples:
        # pad
        pad = clip_samples - len(y)
        y = np.pad(y, (0, pad))
    else:
        # center crop first clip_samples
        y = y[:clip_samples]
    # normalize
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y


def mfcc_features(y, sr=TARGET_SR, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH)
    # mean-variance normalize per-coefficient
    mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-9)
    # aggregate over time (mean + std -> 2*n_mfcc)
    mean = mfcc.mean(axis=1)
    std = mfcc.std(axis=1)
    return np.concatenate([mean, std], axis=0).astype(np.float32)  # shape (2*N_MFCC,)


def logmel_spectrogram(y, sr=TARGET_SR, n_mels=N_MELS):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=n_mels)
    logmel = librosa.power_to_db(mel + 1e-9)
    # z-norm per-frequency
    logmel = (logmel - logmel.mean(axis=1, keepdims=True)) / (logmel.std(axis=1, keepdims=True) + 1e-9)
    # add channel dim for CNN: (time, freq, 1) -> we will transpose to (freq, time, 1)
    logmel = logmel.astype(np.float32)
    # Keras default is (H, W, C). We'll use (F, T, 1)
    return np.expand_dims(logmel, axis=-1)  # (n_mels, time, 1)


def collect_files(root, class_names):
    files, labels = [], []
    for idx, cname in enumerate(class_names):
        for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg"):
            files.extend(glob.glob(os.path.join(root, cname, ext)))
            labels.extend([idx] * len(glob.glob(os.path.join(root, cname, ext))))
    return files, np.array(labels, dtype=np.int64)


# -------------------------------
# Models
# -------------------------------
def build_binary_mlp(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_multiclass_cnn(input_shape, n_classes):
    # input_shape = (n_mels, time, 1)
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPool2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPool2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# -------------------------------
# Exporters
# -------------------------------
def save_keras_and_tflite(model, out_path_h5, out_path_tflite, example_input_shape):
    os.makedirs(os.path.dirname(out_path_h5), exist_ok=True)
    model.save(out_path_h5)
    # TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(out_path_tflite, "wb") as f:
        f.write(tflite_model)
    print(f"Saved: {out_path_h5}")
    print(f"Saved: {out_path_tflite}")


# -------------------------------
# Notification stub (replace with FCM/MQTT/etc.)
# -------------------------------
def notify_parent(title, message):
    # TODO: integrate Firebase / APNs / MQTT publish here
    print(f"[NOTIFY] {title}: {message}")


# -------------------------------
# Pipelines
# -------------------------------
def train_binary(data_root, out_dir):
    classes = ["cry", "nocry"]
    files, labels = collect_files(data_root, classes)
    if len(files) < 10:
        raise RuntimeError("Not enough audio files. Add more to dataset/binary/{cry,nocry}/")

    X = []
    for f in files:
        y = load_clip(f)
        X.append(mfcc_features(y))
    X = np.stack(X)
    y = labels.astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=labels)
    model = build_binary_mlp(X.shape[1])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=2)

    # Eval
    y_pred = (model.predict(X_val).ravel() > 0.5).astype(np.int32)
    print("Binary accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred, target_names=classes))

    # Save
    os.makedirs(out_dir, exist_ok=True)
    save_keras_and_tflite(model,
                          os.path.join(out_dir, "binary_mlp.h5"),
                          os.path.join(out_dir, "binary_mlp.tflite"),
                          (X.shape[1],))


def train_multiclass(data_root, out_dir):
    classes = ["hungry", "diaper", "discomfort", "temperature", "unknown"]
    files, labels = collect_files(data_root, classes)
    if len(files) < 25:
        raise RuntimeError("Not enough audio files. Add more to dataset/multiclass/<class>/")

    # Extract log-mel features
    X = []
    for f in files:
        y = load_clip(f)
        lm = logmel_spectrogram(y)  # (n_mels, time, 1)
        X.append(lm)
    X = np.stack(X)  # (N, n_mels, time, 1)
    y = labels

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=labels)
    input_shape = X.shape[1:]  # (n_mels, time, 1)
    model = build_multiclass_cnn(input_shape, n_classes=len(classes))
    model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_val, y_val), verbose=2)

    # Eval
    y_pred = model.predict(X_val).argmax(axis=1)
    print("Multiclass accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred, target_names=classes))

    # Save
    os.makedirs(out_dir, exist_ok=True)
    save_keras_and_tflite(model,
                          os.path.join(out_dir, "mclass_cnn.h5"),
                          os.path.join(out_dir, "mclass_cnn.tflite"),
                          input_shape)


def infer_binary(model_path, audio_file):
    model = tf.keras.models.load_model(model_path)
    y = load_clip(audio_file)
    feat = mfcc_features(y)[None, ...]
    prob = float(model.predict(feat)[0][0])
    cry = prob > 0.5
    if cry:
        notify_parent("Baby Crying", f"Your baby is crying! (confidence {prob:.2f})")
    else:
        print(f"Not crying (confidence {1-prob:.2f})")


def infer_multiclass(model_path, audio_file):
    classes = ["Hungry", "Diaper", "Discomfort", "Temperature", "Unknown"]
    model = tf.keras.models.load_model(model_path)
    y = load_clip(audio_file)
    lm = logmel_spectrogram(y)[None, ...]
    probs = model.predict(lm)[0]
    top = int(np.argmax(probs))
    msg = f"Your baby is crying — likely {classes[top]} (confidence {probs[top]:.2f})."
    notify_parent("Baby Cry Reason", msg)


# -------------------------------
# CLI
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Baby Cry ML Pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("train-binary")
    p1.add_argument("--data", required=True, help="path to dataset/binary")
    p1.add_argument("--out", required=True, help="output dir for models")

    p2 = sub.add_parser("train-multiclass")
    p2.add_argument("--data", required=True, help="path to dataset/multiclass")
    p2.add_argument("--out", required=True, help="output dir for models")

    p3 = sub.add_parser("infer-binary")
    p3.add_argument("--model", required=True, help="path to binary .h5")
    p3.add_argument("--file", required=True, help="audio file to test")

    p4 = sub.add_parser("infer-multiclass")
    p4.add_argument("--model", required=True, help="path to multiclass .h5")
    p4.add_argument("--file", required=True, help="audio file to test")

    args = parser.parse_args()

    if args.cmd == "train-binary":
        train_binary(args.data, args.out)
    elif args.cmd == "train-multiclass":
        train_multiclass(args.data, args.out)
    elif args.cmd == "infer-binary":
        infer_binary(args.model, args.file)
    elif args.cmd == "infer-multiclass":
        infer_multiclass(args.model, args.file)


if __name__ == "__main__":
    main()
