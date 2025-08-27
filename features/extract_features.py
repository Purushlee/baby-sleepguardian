import librosa
import numpy as np

def extract_features(file_path, n_mfcc=13):
    """
    Extract MFCC features from an audio file.

    Parameters:
    - file_path: str, path to the audio file
    - n_mfcc: int, number of MFCC features to extract (default 13)

    Returns:
    - numpy array of shape (n_mfcc,), mean MFCCs across time frames
    """
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=None)
        
        # Compute MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        # Take the mean of each MFCC across time frames
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        return mfccs_mean

    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return np.zeros(n_mfcc)  # return zeros if there is an error
