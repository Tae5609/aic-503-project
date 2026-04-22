# Image Captioning with EfficientNet + Transformer

Image captioning model for food images. The encoder uses a pretrained
**EfficientNet** as a feature extractor stacked on top of Transformer encoder
layers, and the decoder is a Transformer that generates captions tokenised with
a BERT tokenizer (`bert-base-uncased`).

This directory contains everything needed to train from scratch, finetune from a
pretrained checkpoint, generate captions for a test set, and evaluate the
generated captions with BLEU, BERTScore, and CIDEr.

Note: this directory was cloned from tranquoctrinh/Image-Captioning-EfficientNet-Transformer and been modify

---

## 1. Setup

### Requirements

- Python 3.10+
- An NVIDIA GPU with CUDA 12.8 (the pinned `torch==2.11.0+cu128` wheel in
  `requirements.txt` targets CUDA 12.8 — adjust the `torch` / `torchvision`
  versions if your system uses a different CUDA toolkit).

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

NLTK needs the tokenizer data for BLEU computation:

```bash
python -c "import nltk; nltk.download('punkt')"
```

---

## 2. Expected data layout

All scripts assume the dataset lives one level up from this folder, matching the
project's `dataset/` directory:

```
aic-503-project/
├── dataset/
│   ├── train/              # unzip train.zip here
│   ├── val/                # unzip val.zip here
│   ├── test/               # unzip test.zip here
│   ├── train_label.csv     # columns: filename, caption
│   ├── val_label.csv
│   └── test_label.csv
└── image-captioning-efficientnet/
```

Unzip the dataset archives before running anything:

```bash
cd ../dataset
unzip train.zip -d train
unzip val.zip   -d val
unzip test.zip  -d test
cd ../image-captioning-efficientnet
```

Each label CSV must have two columns:

| filename     | caption                           |
| ------------ | --------------------------------- |
| `000001.jpg` | `a bowl of green curry with rice` |

---

## 3. Exploratory data analysis (optional)

Plots the top-20 most frequent words across training captions and saves
`word_frequency.png`:

```bash
python eda.py
```

---

## 4. Training from scratch

`train.py` trains the full encoder-decoder on `train_label.csv` /
`val_label.csv`, saves the best checkpoint (by validation BLEU-4), and writes a
training log.

```bash
python train.py \
    --train_csv       ../dataset/train_label.csv \
    --val_csv         ../dataset/val_label.csv \
    --train_image_dir ../dataset/train \
    --val_image_dir   ../dataset/val \
    --batch_size      16 \
    --n_epochs        25 \
    --learning_rate   1e-4 \
    --early_stopping  5 \
    --model_path      ./checkpoints/baseline.pt \
    --log_path        ./checkpoints/baseline_log.json \
    --log_visualize_dir ./images/ \
    --device          cuda
```

Useful model flags (defaults shown):

| flag               | default             | description                      |
| ------------------ | ------------------- | -------------------------------- |
| `--embedding_dim`  | `512`               | transformer embedding size       |
| `--encoder_layers` | `6`                 | extra transformer encoder layers |
| `--decoder_layers` | `12`                | transformer decoder layers       |
| `--num_heads`      | `8`                 | attention heads                  |
| `--max_seq_len`    | `128`               | max caption length               |
| `--tokenizer`      | `bert-base-uncased` | HF tokenizer name                |
| `--dropout`        | `0.1`               |                                  |

The first epoch is slow because `datasets.py` lazily caches preprocessed image
tensors as `.pt` files next to the `.jpg`s. Subsequent epochs reuse the cache.

---

## 5. Generating captions for the test set

`generate.py` loads a trained checkpoint, runs beam-search decoding on every
image listed in an input CSV, and writes the predictions to an output CSV with
columns `filename, caption, predicted_caption`.

Configuration lives at the top of `generate.py` — edit the `MODEL_CONFIG` dict
and path constants to match the checkpoint you want to use, then run:

```bash
python generate.py
```

Defaults:

- `IMAGE_FOLDER = "../dataset/test"`
- `CSV_INPUT_PATH = "../dataset/test_label.csv"`
- `CSV_OUTPUT_PATH = "./generated/improvement.csv"`
- `model_path = "./checkpoints/improvement.pt"`
- `beam_size = 3`

Make sure the `encoder_layers`, `decoder_layers`, `embedding_dim`, `num_heads`,
`max_seq_len`, and `dropout` in `MODEL_CONFIG` match whatever was used at
training time — otherwise `load_state_dict` will fail with shape mismatches.

Results are flushed to disk every `SAVE_EVERY_N` images so a crash doesn't lose
progress.

---

## 6. Evaluating generated captions

`evaluate_captions.py` takes the CSV produced by `generate.py` and reports:

- **BLEU-1 through BLEU-4** (corpus-level, NLTK)
- **BERTScore** precision / recall / F1
- **CIDEr** (pycocoevalcap)

```bash
python evaluate_captions.py --csv ./generated/improvement.csv
# or pick the device explicitly for BERTScore:
python evaluate_captions.py --csv ./generated/improvement.csv --device cuda
```

A per-image breakdown is saved next to the input CSV as
`<name>_scores.csv`.

---

## 7. File overview

| file                   | purpose                                                          |
| ---------------------- | ---------------------------------------------------------------- |
| `models.py`            | EfficientNet encoder + Transformer encoder/decoder definitions   |
| `datasets.py`          | `ImageCaptionDataset` — reads the CSV, caches image tensors      |
| `utils.py`             | image transforms, training-log plots, COCO metric helpers        |
| `train.py`             | train from scratch                                               |
| `generate.py`          | beam-search caption generation to CSV                            |
| `evaluate_captions.py` | BLEU / BERTScore / CIDEr from a predictions CSV                  |
| `evaluation.py`        | COCO-Karpathy evaluation pipeline                                |
| `eda.py`               | quick word-frequency plot of training captions                   |
| `generated/`           | prediction CSVs from the baseline / pretrained / improved models |
| `requirements.txt`     | pinned Python dependencies                                       |
