import librosa
import numpy as np

def extract_features(file_path, mfcc=True, mel=True, n_mfcc=13):
    """
    Extract audio features from a given file.
    Args:
        file_path (str): path to audio file
        mfcc (bool): whether to extract MFCC features
        mel (bool): whether to extract Mel-spectrogram features
        n_mfcc (int): number of MFCC coefficients
    Returns:
        np.ndarray: feature vector
    """
    y, sr = librosa.load(file_path, sr=None)
    features = []

    if mfcc:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        features.extend(mfccs_mean)

    if mel:
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_mean = np.mean(mel_spec_db.T, axis=0)
        features.extend(mel_mean)

    return np.array(features)
