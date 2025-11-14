#!/usr/bin/env python
"""
Transform precomputed SigLIP glyph embeddings using a trained MLP head.

- Loads hyperparameters (hidden_dim, output_dim) from validation_metrics.json
- Rebuilds the MLP head, loads weights from siglip_mlp_head_best.pt
- Applies the head (with L2-normalization) to all embeddings from:
    data/embeddings/siglip_glyphs/image_embeddings_{train,validate,test}.npz
- Saves transformed embeddings as dicts into:
    data/embeddings/ft_siglip_glyphs/image_embeddings_{train,validate,test}.npz
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


# -----------------------------
# Model definition
# -----------------------------


class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# -----------------------------
# Utilities
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Transform precomputed SigLIP glyph embeddings using a trained "
            "MLP head and save L2-normalized head embeddings."
        )
    )

    parser.add_argument(
        "--head-weights",
        type=str,
        default="outputs/ft_image_encoder/siglip_mlp_head_best.pt",
        help="Path to trained MLP head weights (.pt).",
    )
    parser.add_argument(
        "--metrics-json",
        type=str,
        default="outputs/ft_image_encoder/validation_metrics.json",
        help="Path to validation_metrics.json with hyperparams.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/embeddings/siglip_glyphs",
        help="Directory containing original SigLIP embedding .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/embeddings/ft_siglip_glyphs",
        help="Directory to save transformed embeddings.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        choices=["train", "val", "test"],
        default=["train", "val", "test"],
        help="Which splits to transform. Default: train val test",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for batched forward passes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'cuda', 'cpu', or 'auto' (default: auto).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing output .npz files for a split.",
    )

    return parser.parse_args()


def load_hyperparams(metrics_json_path):
    if not os.path.isfile(metrics_json_path):
        raise FileNotFoundError(
            f"metrics-json file not found: {metrics_json_path}"
        )

    with open(metrics_json_path, "r") as f:
        data = json.load(f)

    hyper = data.get("hyperparams", None)
    if hyper is None:
        raise KeyError(
            f"'hyperparams' not found in {metrics_json_path}. "
            "Expected keys: 'hidden_dim', 'output_dim', etc."
        )

    if "hidden_dim" not in hyper or "output_dim" not in hyper:
        raise KeyError(
            f"'hidden_dim' and/or 'output_dim' missing in hyperparams of {metrics_json_path}."
        )

    hidden_dim = int(hyper["hidden_dim"])
    output_dim = int(hyper["output_dim"])

    return hidden_dim, output_dim


def load_first_embedding_dim(npz_path):
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Embedding file not found: {npz_path}")

    npz = np.load(npz_path, allow_pickle=True)
    keys = list(npz.keys())
    if not keys:
        raise ValueError(f"No embeddings found in {npz_path}.")

    first_vec = np.asarray(npz[keys[0]], dtype=np.float32).reshape(-1)
    input_dim = first_vec.shape[0]
    return input_dim


def maybe_strip_module_prefix(state_dict):
    """
    If the state_dict keys are prefixed with 'module.' (from DataParallel),
    strip that prefix so it matches the plain MLPHead.
    """
    if not state_dict:
        return state_dict

    needs_strip = all(k.startswith("module.") for k in state_dict.keys())
    if not needs_strip:
        return state_dict

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[len("module.") :]
        new_state_dict[new_key] = v
    return new_state_dict


def build_model(head_weights_path, input_dim, hidden_dim, output_dim, device):
    if not os.path.isfile(head_weights_path):
        raise FileNotFoundError(f"Head weights file not found: {head_weights_path}")

    model = MLPHead(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    state_dict = torch.load(head_weights_path, map_location="cpu")
    state_dict = maybe_strip_module_prefix(state_dict)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def split_to_filename(split):
    if split == "train":
        return "image_embeddings_train.npz"
    elif split == "val":
        return "image_embeddings_validate.npz"
    elif split == "test":
        return "image_embeddings_test.npz"
    else:
        raise ValueError(f"Unknown split: {split}")


def transform_split(
    split,
    input_dir,
    output_dir,
    model,
    device,
    batch_size=4096,
    overwrite=False,
):
    in_name = split_to_filename(split)
    in_path = os.path.join(input_dir, in_name)
    out_path = os.path.join(output_dir, in_name)

    if not os.path.isfile(in_path):
        print(f"[{split}] Input file not found, skipping: {in_path}")
        return

    if os.path.isfile(out_path) and not overwrite:
        print(f"[{split}] Output file already exists, skipping (use --overwrite to regenerate): {out_path}")
        return

    print(f"\n[{split}] Loading embeddings from: {in_path}")
    npz = np.load(in_path, allow_pickle=True)
    keys = list(npz.keys())
    n = len(keys)
    print(f"[{split}] Found {n} embeddings.")

    os.makedirs(output_dir, exist_ok=True)

    out_dict = {}

    # Batched forward pass
    for start in tqdm(range(0, n, batch_size), desc=f"[{split}] Transforming"):
        end = min(start + batch_size, n)
        batch_keys = keys[start:end]

        # Build batch array
        batch_vecs = []
        for k in batch_keys:
            vec = np.asarray(npz[k], dtype=np.float32).reshape(1, -1)
            batch_vecs.append(vec)

        X = np.vstack(batch_vecs)  # shape: (B, input_dim)
        X_t = torch.from_numpy(X).to(device)

        with torch.no_grad():
            Z = model(X_t)  # (B, output_dim)
            Z = nn.functional.normalize(Z, p=2, dim=1)  # L2-normalize

        Z_np = Z.cpu().numpy().astype(np.float32)
        for i, k in enumerate(batch_keys):
            out_dict[k] = Z_np[i]

    print(f"[{split}] Saving transformed embeddings to: {out_path}")
    # Save dict as .npz, keys preserved
    np.savez(out_path, **out_dict)
    print(f"[{split}] Done.")


# -----------------------------
# Main
# -----------------------------


def main():
    args = parse_args()

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load hyperparameters from metrics JSON
    hidden_dim, output_dim = load_hyperparams(args.metrics_json)
    print(f"Loaded hyperparams from {args.metrics_json}:")
    print(f"  hidden_dim = {hidden_dim}")
    print(f"  output_dim = {output_dim}")

    # Determine input_dim from one of the requested splits (prefer train, then val, then test)
    # This assumes all splits share the same embedding dimensionality.
    split_priority = ["train", "val", "test"]
    first_split_for_dim = None
    for s in split_priority:
        if s in args.splits:
            first_split_for_dim = s
            break
    if first_split_for_dim is None:
        raise ValueError("No splits selected to transform (splits list is empty).")

    first_in_name = split_to_filename(first_split_for_dim)
    first_npz_path = os.path.join(args.input_dir, first_in_name)
    input_dim = load_first_embedding_dim(first_npz_path)
    print(f"Detected input embedding dimension from {first_npz_path}: {input_dim}")

    # Build model and load weights
    model = build_model(
        head_weights_path=args.head_weights,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        device=device,
    )
    print(f"Loaded MLP head from: {args.head_weights}")

    # Transform requested splits
    for split in args.splits:
        transform_split(
            split=split,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            model=model,
            device=device,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )

    print("\nAll requested splits transformed.")


if __name__ == "__main__":
    main()

# run this 
# (can add train val and or test to splits argument - currently it just transforms val)

# python ft_image_encoder/transform_embeddings.py \
#   --head-weights outputs/ft_image_encoder/siglip_mlp_head_best.pt \
#   --metrics-json outputs/ft_image_encoder/validation_metrics.json \
#   --input-dir   data/embeddings/siglip_glyphs \
#   --output-dir  data/embeddings/ft_siglip_glyphs \
#   --batch-size 4096 \
#   --device auto \
#   --splits val