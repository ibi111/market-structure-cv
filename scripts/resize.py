from PIL import Image
import os

input_folder = "data/raw-data/5min"
output_folder = "data/data-resized/5min"
target_size = (1280, 800)

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        img = Image.open(f"{input_folder}/{filename}")
        img = img.resize(target_size, Image.LANCZOS)
        img.save(f"{output_folder}/{filename}")