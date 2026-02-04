"""
Download sample medicine images for testing
"""

import requests
from pathlib import Path

# Create input directory
input_dir = Path("data/input")
input_dir.mkdir(parents=True, exist_ok=True)

# Sample medicine images (public domain/open source)
sample_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Ibuprofen-3D-balls.png/220px-Ibuprofen-3D-balls.png",
]

print("Downloading sample images...")

for idx, url in enumerate(sample_urls):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            output_path = input_dir / f"sample_medicine_{idx+1}.png"
            output_path.write_bytes(response.content)
            print(f"✓ Downloaded: {output_path}")
    except Exception as e:
        print(f"✗ Failed to download: {e}")

print("\nYou can also manually add medicine label images to data/input/")
