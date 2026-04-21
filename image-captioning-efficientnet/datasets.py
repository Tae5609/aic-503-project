"""
This file was updated by Shine on April 20, 2026 to support CSV-based image-caption dataset instead of COCO format and enable custom fine-tuning with separate train and validation data.
Changes:
1. Removed manual [CLS]/[SEP] (BERT handles it internally)
2. Added text normalization (whitespace + lowercase cleaning)
3. Removed unnecessary duplicate caption structure
4. Cleaned dataset pipeline for stable fine-tuning
"""

import pandas as pd
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm

from utils import transform


# =========================
# Image preprocessing cache
# =========================
def create_image_inputs(image_dir, transform):
    """
    Precompute image tensors for faster training.
    """
    for root, _, files in os.walk(image_dir):
        for file in tqdm(files):
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)

                image = Image.open(image_path).convert("RGB")
                image = transform(image)

                torch.save(image, image_path.replace(".jpg", ".pt"))


# =========================
# Dataset
# =========================
class ImageCaptionDataset(Dataset):
    def __init__(self, csv_path, image_dir, tokenizer, max_seq_len=128, transform=None):
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.transform = transform

        self.df = self.create_inputs()

    # -------------------------
    # Load + clean CSV
    # -------------------------
    def create_inputs(self):
        df = pd.read_csv(self.csv_path)

        rows = []

        for idx, row in df.iterrows():
            caption = self.normalize_text(row["caption"])

            rows.append({
                "image_id": idx,
                "image_path": os.path.join(self.image_dir, row["filename"]),
                "caption": caption
            })

        return pd.DataFrame(rows)

    # -------------------------
    # Text preprocessing
    # -------------------------
    def normalize_text(self, text):
        """
        Clean caption text:
        - lowercase
        - remove extra spaces
        """
        text = str(text).lower()
        text = " ".join(text.split())   # removes extra whitespace
        return text

    def __len__(self):
        return len(self.df)

    # -------------------------
    # Get item
    # -------------------------
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = row["image_path"]

        # Load cached image tensor if exists
        image_torch_path = image_path.replace(".jpg", ".pt")

        if os.path.exists(image_torch_path):
            image = torch.load(image_torch_path)
        else:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            torch.save(image, image_torch_path)

        caption = row["caption"]

        # -------------------------
        # FIX: NO manual [CLS]/[SEP]
        # -------------------------
        caption_tokens = self.tokenizer(
            caption,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )["input_ids"][0]

        return {
            "image_id": row["image_id"],
            "image_path": image_path,
            "image": image,
            "caption": caption_tokens
        }


# =========================
# Debug run
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased")
    parser.add_argument("--max_seq_len", type=int, default=128)

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    dataset = ImageCaptionDataset(
        csv_path=args.csv_path,
        image_dir=args.image_dir,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        transform=transform
    )

    print(dataset[0])

    create_image_inputs(args.image_dir, transform)