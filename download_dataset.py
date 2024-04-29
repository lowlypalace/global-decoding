import os
import requests
from tqdm import tqdm

def download_dataset(subdir="data", datasets=["webtext"], splits=["train", "valid", "test"], base_url="https://openaipublic.azureedge.net/gpt-2/output-dataset/v1/"):
    """Download datasets from a specified URL and save them into a local directory."""
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    subdir = subdir.replace("\\", "/")  # Normalize path for Windows compatibility

    for ds in datasets:
        for split in splits:
            filename = f"{ds}.{split}.jsonl"
            file_path = os.path.join(subdir, filename)
            url = f"{base_url}{filename}"

            # Check if file already exists to potentially skip download
            if os.path.exists(file_path):
                print(f"{filename} already exists. Skipping download.")
                continue

            # Start download
            print(f"Starting download of {filename}")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Check that the request was successful

                with open(file_path, "wb") as f:
                    file_size = int(response.headers.get("content-length", 0))
                    chunk_size = 1024  # Commonly used size for network chunk downloads
                    progress = tqdm(total=file_size, unit="iB", unit_scale=True, desc=f"Downloading {filename}")
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        size = f.write(chunk)
                        progress.update(size)
                    progress.close()
            except requests.RequestException as e:
                print(f"Failed to download {filename}: {e}")
                if os.path.exists(file_path):
                    os.remove(file_path)  # Remove partial files on failure

if __name__ == "__main__":
    download_dataset()
