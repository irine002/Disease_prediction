import requests
import os

url = "https://raw.githubusercontent.com/anujdutt9/Disease-Prediction-from-Symptoms/master/dataset/training_data.csv"
output_path = "dataset.csv"

def download_data():
    print(f"Downloading data from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, "w", encoding='utf-8') as f:
            f.write(response.text)
        print(f"Dataset saved to {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    download_data()
