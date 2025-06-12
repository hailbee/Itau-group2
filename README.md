# 🕵️‍♂️ Spoof Detection with CLIP Embeddings - Itaú Group 2

Fraudsters are creating visually similar spoof accounts to impersonate trusted companies, posing a serious risk to financial institutions like Itaú Unibanco. This repository contains the code and data used to train, test, and evaluate a spoof detection system based on CLIP embeddings, combined with Cosine and Euclidean similarity metrics. Our approach begins by training the model on a dataset of spoofed names, then testing it on a ~1,800-name German dataset. We evaluate performance using confusion matrices, accuracy, and precision.

---

## 📁 Repository Structure


---

## 🧾 Data Overview

- `data/raw/`: Contains the original testing dataset of ~1,800 German company names — including those with special characters.
- `data/processed/`: Includes cleaned intermediate and final CSV/PKL files:
  - `merged_data.pkl`: Fully processed pairwise data (`name1`, `name2`, `label`) used for contrastive loss training.
  - `fraud_triplets.pkl`: Fully processed triplet data (`fraud_name`, `real_name`, `negative_name`) used for triplet loss training.
  - `german_merged_dataset.csv`: Testing data containing normalized German company name and label.
  - `german_companies_after_500.csv`: Reference set containing real German company names used for testing.
    - Spoof versions of these names are integrated into `german_merged_dataset.csv`.

---

## 📓 Notebook Breakdown

- `notebooks/full_run_v2/`: Iteratively tests different parameters for both pairwise/triplet datsets depending on function input.
- `notebooks/test_raw_clip/`: Testing the effectiveness of pre-trained CLIP embeddings on identifying spoof.

---

## 🧠 Models

- `models/models.py`: Contains different variations of the SiameseCLIP model:
  - `BaseSiameseCLIP`: Core model that wraps a frozen or trainable CLIP encoder and a 2-layer projector to produce normalized text embeddings.
  - `SiameseCLIPModelPairs`: Extends `BaseSiameseCLIP`; returns embeddings for a pair of inputs, used with contrastive loss.
  - `SiameseCLIPTriplet`: Extends `BaseSiameseCLIP`; returns embeddings for anchor, positive, and negative inputs, used with triplet loss.

---

## ⚙️ Scripts

- `scripts/grid_search.py`: Performs grid search across hyperparameters (e.g., learning rate, batch size, margin) to train and evaluate Siamese models using either contrastive or triplet loss.
- `scripts/train.py`: Contains training loops for both contrastive (`train_pair`) and triplet (`train_triplet`) loss models.
- `scripts/test.py`: Computes model predictions by comparing test names to a reference set using Cosine/Euclidean similarity of projected embeddings.
- `scripts/eval.py`: Evaluates model predictions using ROC curve, confusion matrix, and threshold-based accuracy metrics.
- 'scripts/test_raw_clip.py': Evaluates performance using raw (untrained) CLIP embeddings for baseline comparison.

---

## 🛠️ Utils
- `utils/embeddings.py`: Defines helper functions for generating CLIP-based embeddings, including `EmbeddingExtractor`, `batched_embedding`.
- `utils/loss.py`: Defines multiple loss functions including contrastive, triplet (Cosine & Euclidean), and hybrid variations.
- `utils/data.py`: Contains PyTorch dataset classes for handling contrastive and triplet data formats.
- `utils/evals.py`: Provides plotting utilities and threshold optimization functions for model evaluation.

## 📦 Installation

We recommend using a Python 3.10 virtual environment.

```bash
git clone https://github.com/sashajh2/itau-group2.git
cd itau-group2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
