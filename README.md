# Spoof Detection with CLIP Embeddings - Itaú Group 2

Fraudsters are creating visually similar spoof accounts to impersonate trusted companies, posing a serious risk to financial institutions like Itaú Unibanco. This repository contains the code and data used to train, test, and evaluate different spoof detection systems based on Cosine and Euclidean similarity metrics. We evaluate performance using confusion matrices, accuracy, and precision.

We recommend using a Python >= 3.10 virtual environment.

# Example experiment options

* Current Pipeline (name pairs -> precomputed glyph embeddings using SigLip vision model -> similarity)

```bash
python .scripts/main.py --backbone precomputed --pairs data/processed/validate_pairs_ref_10k.parquet --left-col fraudulent_name --right-col real_name --label-col label --npz data/embeddings/siglip_glyphs/image_embeddings_validate.npz --mode baseline --pc-non-strict
```

* VA-TE Pipeline (name pairs -> text embeddings from SigLip text model -> transformed embeddings from projection head -> similarity)

```bash
python .scripts/main.py --backbone siglip --pairs data/processed/validate_pairs_ref_10k.parquet --left-col fraudulent_name --right-col real_name --label-col label --mode baseline --batch-size 32 --model-name google/siglip-base-patch16-224
```

# Cloning the repository and setting up a vritual environment

mac
```bash
git clone https://github.com/hailbee/Itau-group2.git
cd Itau-group2
python3 -m venv itau_env
source itau_env/bin/activate
pip install -r requirements.txt
```

Windows
```bash
git clone https://github.com/hailbee/Itau-group2.git
cd Itau-group2
python -m venv .itau_env
.\itau_env\Scripts\activate
pip install -r requirements.txt
```
