"""
This file was created by Suppakorn (Tae) on April 9, 2026 (using the assistant of Claude) to generate captions and save the result as csv file.
"""

import torch
import time
from datetime import timedelta
from transformers import BertTokenizer
import os
import csv
from tqdm import tqdm

from utils import transform
from models import ImageCaptionModel
from evaluation import generate_caption

# ==========================================
# CONFIGURATION
# ==========================================
IMAGE_FOLDER = "../dataset/test"

# CSV Configuration
CSV_INPUT_PATH = "../dataset/test_label.csv"        # Input CSV with columns: filename, caption
CSV_OUTPUT_PATH = "./generated/improvement.csv"        # Output CSV with columns: filename, caption, predicted_caption
SAVE_EVERY_N = 5                        # Flush results to CSV every N entries
 
# Model Configuration
MODEL_CONFIG = {
    "embedding_dim": 512,
    "tokenizer": "bert-base-uncased",
    "max_seq_len": 128,
    "encoder_layers": 10,
    "decoder_layers": 16,
    "num_heads": 8,
    "dropout": 0.1,
    "model_path": "./checkpoints/improvement.pt",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "beam_size": 3,
}

# ==========================================
def load_captions_from_csv(csv_path: str) -> dict:
    """Load ground-truth captions from a CSV file with columns: filename, caption."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found at: {csv_path}")
 
    captions = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            captions[row["filename"]] = row["caption"]
 
    print(f"Loaded {len(captions)} entries from {csv_path}")
    return captions


def main():
    # Load ground-truth captions from CSV
    captions_map = load_captions_from_csv(CSV_INPUT_PATH)
 
    # Build image list from CSV filenames
    images_to_caption = [
        (filename, os.path.join(IMAGE_FOLDER, filename), caption)
        for filename, caption in captions_map.items()
    ]
    
    # Load model and tokenizer
    device = torch.device(MODEL_CONFIG["device"])
    print(f"Using device: {device}")
 
    tokenizer = BertTokenizer.from_pretrained(MODEL_CONFIG["tokenizer"])
 
    model_params = {
        "embedding_dim": MODEL_CONFIG["embedding_dim"],
        "vocab_size": tokenizer.vocab_size,
        "max_seq_len": MODEL_CONFIG["max_seq_len"],
        "encoder_layers": MODEL_CONFIG["encoder_layers"],
        "decoder_layers": MODEL_CONFIG["decoder_layers"],
        "num_heads": MODEL_CONFIG["num_heads"],
        "dropout": MODEL_CONFIG["dropout"],
    }
 
    # Load model
    print("Loading model...")
    start_time = time.time()
    model = ImageCaptionModel(**model_params)
 
    if not os.path.exists(MODEL_CONFIG["model_path"]):
        print(f"Error: Model file not found at {MODEL_CONFIG['model_path']}")
        return
 
    model.load_state_dict(torch.load(MODEL_CONFIG["model_path"], map_location=device))
    model.to(device)
    model.eval()
 
    time_load_model = str(timedelta(seconds=int(time.time() - start_time)))
    print(f"Done loading model in {time_load_model}")
    print("-" * 50)
 
    # Generate captions and write results to CSV incrementally
    fieldnames = ["filename", "caption", "predicted_caption"]
    results = []
    total_saved = 0
    all_references = []
    all_hypotheses = []
 
    def flush_results(rows: list, output_path: str, write_header: bool) -> None:
        with open(output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
 
    # Clear the output file and write header upfront
    flush_results([], CSV_OUTPUT_PATH, write_header=True)
 
    pbar = tqdm(images_to_caption, desc="Captioning", unit="img", position=0, leave=True)
    for filename, image_path, ground_truth_caption in pbar:
        if not os.path.exists(image_path):
            tqdm.write(f"Skipping: File not found - {image_path}")
            continue
 
        tqdm.write(f"Generating caption for: {filename}")
        st = time.time()
        try:
            predicted_caption = generate_caption(
                model=model,
                image_path=image_path,
                transform=transform,
                tokenizer=tokenizer,
                max_seq_len=MODEL_CONFIG["max_seq_len"],
                beam_size=MODEL_CONFIG["beam_size"],
                device=device,
                print_process=False,
            )

            elapsed = time.time() - st
            tqdm.write(f"--- Caption:    {ground_truth_caption}")
            tqdm.write(f"--- Predicted:  {predicted_caption}")
            tqdm.write(f"--- Time: {elapsed:.2f}s")
            results.append({
                "filename": filename,
                "caption": ground_truth_caption,
                "predicted_caption": predicted_caption,
            })
            all_references.append(ground_truth_caption)
            all_hypotheses.append(predicted_caption)
            pbar.set_postfix({"last": filename, "saved": total_saved})
        except Exception as e:
            tqdm.write(f"--- Error processing {image_path}: {e}")
        tqdm.write("-" * 50)
 
        # Flush every N completed entries
        if len(results) >= SAVE_EVERY_N:
            flush_results(results, CSV_OUTPUT_PATH, write_header=False)
            total_saved += len(results)
            tqdm.write(f"[Checkpoint] Saved {total_saved} entries to {CSV_OUTPUT_PATH}")
            pbar.set_postfix({"last": filename, "saved": total_saved})
            results.clear()
 
    # Flush any remaining entries
    if results:
        flush_results(results, CSV_OUTPUT_PATH, write_header=False)
        total_saved += len(results)
 
    print(f"\nDone. Saved {total_saved} results to {CSV_OUTPUT_PATH}")
 
 
if __name__ == "__main__":
    main()
