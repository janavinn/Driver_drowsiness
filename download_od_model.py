import urllib.request
import os

URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
FILE_NAME = "efficientdet_lite0.tflite"

def download_model():
    if os.path.exists(FILE_NAME):
        print(f"{FILE_NAME} already exists.")
        return

    print(f"Downloading {FILE_NAME}...")
    try:
        urllib.request.urlretrieve(URL, FILE_NAME)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download: {e}")

if __name__ == "__main__":
    download_model()
