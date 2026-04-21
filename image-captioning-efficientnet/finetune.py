"""
Finetune script

Same pipeline as train.py, but initializes the model from a pretrained
checkpoint (default: pretrained/model_image_captioning_eff_transfomer.pt)
before continuing training.
"""

import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import os
import json

from utils import transform, visualize_log
from datasets import ImageCaptionDataset
from models import ImageCaptionModel

smoothie = SmoothingFunction()


# =========================
# BLEU helper
# =========================
def compute_bleu_scores(references, hypotheses):
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0), smoothing_function=smoothie.method4)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie.method4)
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie.method4)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie.method4)

    return bleu1, bleu2, bleu3, bleu4


# =========================
# TRAIN EPOCH
# =========================
def train_epoch(model, loader, tokenizer, criterion, optimizer, epoch, device):
    model.train()

    total_loss = []
    hypotheses, references = [], []

    bar = tqdm(loader, desc=f"Finetune epoch {epoch+1}")

    for batch in bar:
        image = batch["image"].to(device)
        caption = batch["caption"].to(device)

        target_input = caption[:, :-1]
        preds = model(image, target_input)

        optimizer.zero_grad()

        gold = caption[:, 1:].contiguous().view(-1)
        loss = criterion(preds.view(-1, preds.size(-1)), gold)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

        preds = torch.argmax(F.softmax(preds, dim=-1), dim=-1)
        preds = preds.detach().cpu().numpy()

        decoded_preds = [tokenizer.decode(p, skip_special_tokens=True) for p in preds]
        hypo = [p.split() for p in decoded_preds]

        refs = [
            [tokenizer.decode(caption[j], skip_special_tokens=True).split()]
            for j in range(caption.size(0))
        ]

        hypotheses += hypo
        references += refs

        bar.set_postfix(loss=loss.item())

    bleu1, bleu2, bleu3, bleu4 = compute_bleu_scores(references, hypotheses)
    avg_loss = sum(total_loss) / len(total_loss)

    return avg_loss, bleu1, bleu2, bleu3, bleu4


# =========================
# VALIDATION
# =========================
def validate_epoch(model, loader, tokenizer, criterion, epoch, device):
    model.eval()

    total_loss = []
    hypotheses, references = [], []

    with torch.no_grad():
        bar = tqdm(loader, desc=f"Validating epoch {epoch+1}")

        for batch in bar:
            image = batch["image"].to(device)
            caption = batch["caption"].to(device)

            target_input = caption[:, :-1]
            preds = model(image, target_input)

            gold = caption[:, 1:].contiguous().view(-1)
            loss = criterion(preds.view(-1, preds.size(-1)), gold)

            total_loss.append(loss.item())

            preds = torch.argmax(F.softmax(preds, dim=-1), dim=-1)
            preds = preds.detach().cpu().numpy()

            decoded_preds = [tokenizer.decode(p, skip_special_tokens=True) for p in preds]
            hypo = [p.split() for p in decoded_preds]

            refs = [
                [tokenizer.decode(caption[j], skip_special_tokens=True).split()]
                for j in range(caption.size(0))
            ]

            hypotheses += hypo
            references += refs

            bar.set_postfix(loss=loss.item())

    bleu1, bleu2, bleu3, bleu4 = compute_bleu_scores(references, hypotheses)
    avg_loss = sum(total_loss) / len(total_loss)

    return avg_loss, bleu1, bleu2, bleu3, bleu4


# =========================
# TRAIN LOOP
# =========================
def train(model, train_loader, val_loader, optimizer, criterion,
          n_epochs, tokenizer, device, model_path, log_path, early_stopping):

    log = {
        "train_loss": [],
        "val_loss": [],
        "train_bleu": [],
        "val_bleu": []
    }

    best_bleu4 = -np.inf
    patience = 0
    start_time = time.time()

    for epoch in range(n_epochs):

        train_loss, t1, t2, t3, t4 = train_epoch(
            model, train_loader, tokenizer, criterion, optimizer, epoch, device
        )

        val_loss, v1, v2, v3, v4 = validate_epoch(
            model, val_loader, tokenizer, criterion, epoch, device
        )

        if v4 > best_bleu4:
            best_bleu4 = v4
            torch.save(model.state_dict(), model_path)
            patience = 0
            print("Saved best model (BLEU-4 improved)")
        else:
            patience += 1
            if patience >= early_stopping:
                print("Early stopping triggered")
                break

        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)
        log["train_bleu"].append([t1, t2, t3, t4])
        log["val_bleu"].append([v1, v2, v3, v4])

        print(
            f"Epoch {epoch+1}/{n_epochs}\n"
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n"
            f"Train BLEU: {t1:.3f}/{t2:.3f}/{t3:.3f}/{t4:.3f}\n"
            f"Val BLEU:   {v1:.3f}/{v2:.3f}/{v3:.3f}/{v4:.3f}\n"
            f"Time: {timedelta(seconds=int(time.time()-start_time))}"
        )

        with open(log_path, "w") as f:
            json.dump(log, f)

    return log


# =========================
# MAIN
# =========================
def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--encoder_layers", type=int, default=3)
    parser.add_argument("--decoder_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--train_image_dir", type=str, required=True)
    parser.add_argument("--val_image_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_epochs", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--early_stopping", type=int, default=5)

    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="./pretrained/model_image_captioning_eff_transfomer.pt",
        help="Path to the pretrained checkpoint used to initialize the model.",
    )
    parser.add_argument("--model_path", type=str, default="./model_finetuned.pt")
    parser.add_argument("--log_path", type=str, default="./log_finetuned.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_visualize_dir", type=str, default="./images/")
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    train_dataset = ImageCaptionDataset(
        args.train_csv, args.train_image_dir, tokenizer, args.max_seq_len, transform
    )

    val_dataset = ImageCaptionDataset(
        args.val_csv, args.val_image_dir, tokenizer, args.max_seq_len, transform
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = ImageCaptionModel(
        args.embedding_dim,
        tokenizer.vocab_size,
        args.max_seq_len,
        args.encoder_layers,
        args.decoder_layers,
        args.num_heads,
        args.dropout
    ).to(device)

    # Load pretrained weights for finetuning
    if not os.path.isfile(args.pretrained_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained_path}")

    state_dict = torch.load(args.pretrained_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights from {args.pretrained_path}")
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    log = train(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        args.n_epochs,
        tokenizer,
        device,
        args.model_path,
        args.log_path,
        args.early_stopping
    )

    visualize_log(log, args.log_visualize_dir)


if __name__ == "__main__":
    main()
