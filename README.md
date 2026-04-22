# AIC-503 Final Project — Food Image Captioning

End-to-end image captioning pipeline for food images, built for AIC-503.
The repository covers every stage of the project:

1. **Labelling** — generate captions for raw food images with Google Gemini.
2. **Verification** — manual QA pass over the generated labels.
3. **Modelling** — train / finetune / evaluate an EfficientNet + Transformer
   captioning model on the curated dataset.

---

## Repository layout

```
aic-503-project/
├── dataset/                         # train / val / test images + CSV labels
│   ├── train.zip, val.zip, test.zip
│   ├── train_label.csv, val_label.csv, test_label.csv
│   └── uncut_labels/
├── labelling-pipeline/              # Gemini-based auto-labelling (stage 1)
│   ├── generate.py
│   ├── rename.py
│   ├── take_samples.ipynb
│   └── README.md
├── label-verification/              # manual QA artefacts (stage 2)
│   ├── Images/
│   └── Label Verification.xlsx
└── image-captioning-efficientnet/   # the captioning model (stage 3)
    ├── train.py, finetune.py
    ├── generate.py, evaluate_captions.py
    ├── evaluation.py, eda.py
    ├── models.py, datasets.py, utils.py
    ├── requirements.txt
    └── README.md
```

Each sub-project has its own README with detailed commands and flags. This
top-level README shows how the pieces fit together.

---

## Prerequisites

- Python 3.10+
- An NVIDIA GPU with CUDA 12.8 (for training / evaluation)
- A Google Gemini API key (only needed if you want to **regenerate** labels)

Clone the repo and change into it:

```bash
git clone <repo-url> aic-503-project
cd aic-503-project
```

---

## 1. Prepare the dataset

The curated dataset is shipped as three zipped folders of images plus three
CSVs. Each CSV has two columns: `filename,caption`.

```bash
cd dataset
unzip train.zip -d train
unzip val.zip   -d val
unzip test.zip  -d test
cd ..
```

After unzipping:

```
dataset/
├── train/                 # image files referenced by train_label.csv
├── val/
├── test/
├── train_label.csv
├── val_label.csv
└── test_label.csv
```

---

## 2. (Optional) Regenerate captions with the labelling pipeline

Skip this step unless you want to label new raw images from scratch — the
committed CSVs already contain verified captions.

```bash
cd labelling-pipeline
pip install google-genai python-dotenv tqdm pydantic

cp .env.example .env                  # then add GOOGLE_API_KEY=...

# For each split (train / val / test):
#   1. edit folder_path in rename.py, then:
python3 rename.py
#   2. edit IMAGE_FOLDER in generate.py, then:
python3 generate.py
```

Captions are written to `labels.csv`. See `labelling-pipeline/README.md` for
full instructions.

---

## 3. (Optional) Label verification

`label-verification/Label Verification.xlsx` tracks the manual QA pass over the
generated captions. `label-verification/Images/` holds the images that were
reviewed. There's nothing to run here — it documents how the CSVs in
`dataset/` were cleaned up before training.

---

## 4. Train and evaluate the captioning model

All model commands run from `image-captioning-efficientnet/`. Install the
pinned dependencies once:

```bash
cd image-captioning-efficientnet
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

### 4a. Train from scratch

```bash
python train.py \
    --train_csv       ../dataset/train_label.csv \
    --val_csv         ../dataset/val_label.csv \
    --train_image_dir ../dataset/train \
    --val_image_dir   ../dataset/val \
    --model_path      ./checkpoints/baseline.pt \
    --log_path        ./checkpoints/baseline_log.json \
    --device          cuda
```

### 4b. Generate captions for the test set

Edit the `MODEL_CONFIG` block at the top of
`image-captioning-efficientnet/generate.py` to point at the checkpoint you
trained (make sure the architecture hyperparameters match), then:

```bash
python generate.py
```

Predictions are written to `./generated/improvement.csv` by default.

### 4c. Score the predictions

```bash
python evaluate_captions.py --csv ./generated/improvement.csv
```

Reports BLEU-1..4, BERTScore (P/R/F1), and CIDEr, and saves a per-image
breakdown to `./generated/improvement_scores.csv`.

See `image-captioning-efficientnet/README.md` for the full flag reference and
additional options (COCO-Karpathy evaluation, EDA, etc.).
