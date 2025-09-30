# 🕵️‍♂️ Spoof Detection with CLIP Embeddings - Itaú Group 2

Fraudsters are creating visually similar spoof accounts to impersonate trusted companies, posing a serious risk to financial institutions like Itaú Unibanco. This repository contains the code and data used to train, test, and evaluate a spoof detection system based on CLIP embeddings, combined with Cosine and Euclidean similarity metrics. Our approach begins by training the model on a dataset of spoofed names, then testing it on a ~1,800-name German dataset. We evaluate performance using confusion matrices, accuracy, and precision.

We recommend using a Python 3.10 virtual environment.

```bash
git clone https://github.com/sashajh2/itau-group2.git
cd itau-group2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
