import os
from inference.predict import predict

def live_demo(directory):
    """
    Run live demo on a folder of audio files
    """
    for file in os.listdir(directory):
        if file.endswith('.wav'):
            file_path = os.path.join(directory, file)
            result = predict(file_path)
            print(f"{file}: {result}")

if __name__ == "__main__":
    folder = input("Enter folder path for live demo: ")
    live_demo(folder)
