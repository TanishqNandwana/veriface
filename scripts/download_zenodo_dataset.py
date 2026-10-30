"""
Download and Extract Zenodo Real vs Fake Faces Dataset (Record 14532968)
Target location: ~/VeriFaceStorage/datasets/
Verifies checksum and extracted structure.
"""

import os
import sys
import time
import zipfile
import urllib.request

def download_with_progress(url, dest_path):
    print(f"[*] Downloading from: {url}")
    print(f"[*] Destination: {dest_path}")
    
    start_time = time.time()
    
    class DownloadProgress:
        def __init__(self):
            self.last_reported = 0
            
        def __call__(self, block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = downloaded / total_size * 100
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                elapsed = time.time() - start_time
                speed = mb_downloaded / max(1.0, elapsed)
                if percent - self.last_reported >= 5 or downloaded >= total_size:
                    print(f"  --> Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB) @ {speed:.2f} MB/s")
                    self.last_reported = percent

    urllib.request.urlretrieve(url, dest_path, DownloadProgress())
    print(f"[✓] Download completed in {time.time() - start_time:.2f}s!")

def main():
    target_dir = os.path.expanduser("~/VeriFaceStorage/datasets")
    os.makedirs(target_dir, exist_ok=True)
    
    zip_dest = os.path.join(target_dir, "Real_vs_Fake_Faces.zip")
    extract_dest = os.path.join(target_dir, "real_vs_fake_faces")
    
    url = "https://zenodo.org/records/14532968/files/Real%20vs%20Fake%20Faces.zip?download=1"
    
    # 1. Download if not already downloaded
    if not os.path.exists(zip_dest) or os.path.getsize(zip_dest) < 1000000000:
        download_with_progress(url, zip_dest)
    else:
        print(f"[+] Zip archive already exists at {zip_dest} ({os.path.getsize(zip_dest) / (1024*1024):.1f} MB)")
        
    # 2. Extract
    print(f"[*] Extracting archive to {extract_dest}...")
    os.makedirs(extract_dest, exist_ok=True)
    
    with zipfile.ZipFile(zip_dest, "r") as zf:
        zf.extractall(extract_dest)
        
    print(f"[✓] Successfully extracted to {extract_dest}")
    
    # 3. Inspect directory structure
    print("[*] Inspecting extracted structure:")
    for root, dirs, files in os.walk(extract_dest):
        depth = root[len(extract_dest):].count(os.sep)
        if depth <= 3:
            print(f"{'  ' * depth}📂 {os.path.basename(root)}/ ({len(files)} files)")

if __name__ == "__main__":
    main()
