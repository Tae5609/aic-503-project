"""
Created by Suppakorn (Tae) on March 26, 2026. To generate captions for food images using Google's model.
"""

import os
from tqdm import tqdm
from google import genai
from google.genai import types
import csv
from dotenv import load_dotenv
import io
from pydantic import BaseModel

# Loading env variables and generating a client
load_dotenv()
client = genai.Client()

# Define model response config
class ImageCaption(BaseModel):
    filename: str
    caption: str

class CaptionList(BaseModel):
    items: list[ImageCaption]

config = types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=CaptionList,
)

# Parameters
MODEL = "gemini-2.5-flash"
IMAGE_FOLDER = "./images-val"
OUTPUT_CSV = "val_label.csv"
BATCH_SIZE = 6


def append_to_csv(data_items):
    """Append CSV rows to labels.csv."""
    file_exists = os.path.exists(OUTPUT_CSV)

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        # We use quoting=csv.QUOTE_ALL to ensure captions are ALWAYS wrapped in ""
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        
        # Write header only if the file is brand new
        if not file_exists:
            writer.writerow(["filename", "caption"])
        
        # Write the data from the Pydantic objects
        for item in data_items:
            writer.writerow([item.filename, item.caption])


def process_image(filenames):
    """Process each batches"""
    # Input contents
    contents = [
        f"""
        Your role is to generate captions for several restaurant dishes.

        **Instruction**
        - You will be given {len(filenames)} images
        - Generate <= 8 words per caption
        - Focus only on food appearance
        - No restaurant names
        """
    ]

    for i, filename in enumerate(filenames):
        with open(f"{IMAGE_FOLDER}/{filename}", "rb") as f:
            image_bytes = f.read()
        
        contents.append(filename)
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
    
    response = client.models.generate_content(model=MODEL, config=config, contents=contents)
    result = response.parsed if hasattr(response, "parsed") else None
    
    if not result:
        print(f"No result for batch: {filenames}")
        return

    append_to_csv(result.items)


def get_processed_files():
    """Read labels.csv and return a set of already processed filenames."""
    if not os.path.exists(OUTPUT_CSV):
        return set()
    
    with open(OUTPUT_CSV, "r") as f:
        reader = csv.DictReader(f)
        return {row["filename"] for row in reader}


def loop_images(batch_size):
    """Loop through all images in a folder in batches and process each one."""

    # Check if folder exists
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Folder not found: {IMAGE_FOLDER}")
        return

    # Get all files in the folder
    image_files = os.listdir(IMAGE_FOLDER)

    if not image_files:
        print("No images found in folder.")
        return
    
    # Skip processed files
    processed = get_processed_files()
    remaining = [f for f in image_files if f not in processed]
    print(f"Total: {len(image_files)} | Done: {len(processed)} | Remaining: {len(remaining)}")

    if not remaining:
        print("All images already processed!")
        return

    # Split into batches
    remaining.sort()
    batches = [remaining[i:i + batch_size] for i in range(0, len(remaining), batch_size)]

    # Loop through batches and call process_image once per batch
    for batch_num, batch in enumerate(tqdm(batches, desc="Batches", unit="batch")):
        # print(f"\nBatch {batch_num + 1}/{len(batches)} ({len(batch)} images)")
        print(f"Processing batch: {batch}")
        process_image(batch)


loop_images(BATCH_SIZE)
