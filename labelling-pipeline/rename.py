"""
Created by Suppakorn (Tae) on April 2, 2026. To rename the filenames into 6-digit format.
"""

import os
from tqdm import tqdm

folder_path = "./images-val"  # change this to your folder path

# Get all jpg files
files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]

# Sort files (important to keep order consistent)
files.sort()

# Rename files with progress bar
for index, filename in enumerate(tqdm(files, desc="Renaming"), start=1):
    new_name = f"{index:06d}.jpg"

    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)

    os.rename(old_path, new_path)

print("Renaming complete.")