import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, filename):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def main():
    # Create directories if they don't exist
    os.makedirs('data/raw/plant_dataset', exist_ok=True)
    
    # Download the dataset
    url = "https://data.mendeley.com/public-files/datasets/tywbtsjrjv/files/d5652a28-c1d8-4b76-97f3-72fb80f94efc/file_downloaded"
    zip_path = 'data/raw/plant_dataset.zip'
    
    print("Downloading dataset...")
    download_file(url, zip_path)
    
    print("\nExtracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('data/raw/plant_dataset')
    
    # Clean up
    os.remove(zip_path)
    print("Dataset preparation completed!")

if __name__ == '__main__':
    main() 