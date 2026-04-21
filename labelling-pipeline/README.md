# Food Image Caption Generator

## Setup

1. Install dependencies:

```bash
pip install google-genai python-dotenv tqdm pydantic
```

2. Create `.env` and add your API key:

```bash
cp .env.example .env
```

```env
GOOGLE_API_KEY=your_api_key_here
```

---

## Folder Structure

```
.
├── generate.py
├── rename.py
├── images-train/
├── images-val/
├── images-test/
├── labels.csv
└── .env
```

---

## Usage

### 1. Prepare images

Take food images from the dataset and put them into:

- `images-train`
- `images-val`
- `images-test`

---

### 2. Process each folder

For **each folder**, do the following:

#### Step 1: Rename images

Edit `rename.py`:

```python
folder_path = "./images-train"  # change to images-val / images-test
```

Run:

```bash
python3 rename.py
```

#### Step 2: Generate captions

Edit `generate.py`:

```python
IMAGE_FOLDER = "./images-train"  # change to images-val / images-test
```

Run:

```bash
python3 generate.py
```

Captions will be saved to `labels.csv`.

---

Repeat the same steps for:

- `images-train`
- `images-val`
- `images-test`
