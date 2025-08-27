import os
from features.extract_features import extract_features

def load_dataset(directory):
    """
    Load dataset and extract features for each audio file.
    Assumes subdirectories = class labels
    """
    X, y = [], []
    for label in os.listdir(directory):
        class_dir = os.path.join(directory, label)
        if not os.path.isdir(class_dir):
            continue
        for file in os.listdir(class_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(class_dir, file)
                features = extract_features(file_path)
                X.append(features)
                y.append(label)
    return X, y
