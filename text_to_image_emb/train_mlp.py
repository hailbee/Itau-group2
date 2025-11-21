#!/usr/bin/env python
import argparse
import json
import os
from copy import deepcopy

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


# -----------------------------
# Data utilities
# -----------------------------


def load_embeddings_npz(path, max_items=None):
    """
    Load embeddings from a .npz file.

    Assumes format:
      - keys are names (e.g. "foo.png") and each value is a 1D embedding
    """
    npz = np.load(path, allow_pickle=True)
    keys = list(npz.keys())
    if max_items is not None:
        keys = keys[:max_items]

    vectors = []
    name_to_idx = {}

    for idx, key in enumerate(
        tqdm(keys, desc=f"Loading embeddings from {os.path.basename(path)}", total=len(keys))
    ):
        vec = np.asarray(npz[key], dtype=np.float32).reshape(-1)  # ensure 1D
        vectors.append(vec)
        name_to_idx[key] = idx

    emb_matrix = np.stack(vectors, axis=0)  # shape: (N, D)
    print(f"{os.path.basename(path)} embeddings shape: {emb_matrix.shape}")
    return emb_matrix, name_to_idx


class TextImagePairDataset(Dataset):
    """
    Dataset of (text_emb, image_emb, label) pairs backed by separate embedding tensors.
    """

    def __init__(
        self,
        parquet_path,
        text_emb_tensor,
        img_emb_tensor,
        text_name_to_idx,
        img_name_to_idx,
        split_name,
        text_col="fraudulent_name",
        image_col="real_name",
        label_col="label",
        image_suffix=".png",
    ):
        """
        Parameters
        ----------
        parquet_path : str
        text_emb_tensor : torch.FloatTensor, shape (N_text, D_text)
        img_emb_tensor : torch.FloatTensor, shape (N_img, D_img)
        text_name_to_idx : dict[str, int]
        img_name_to_idx : dict[str, int]
        split_name : str, for logging
        text_col : which column in parquet is used for text names
        image_col : which column in parquet is used for image names
        label_col : which column contains labels (0/1)
        image_suffix : suffix to append to image name (e.g. ".png")
        """
        self.text_emb = text_emb_tensor
        self.img_emb = img_emb_tensor

        df = pd.read_parquet(parquet_path)
        required_cols = {text_col, image_col, label_col}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{parquet_path} missing columns: {missing}")

        total_pairs = len(df)
        text_indices = []
        img_indices = []
        labels = []

        missing_text = 0
        missing_img = 0
        bad_label = 0

        for _, row in df.iterrows():
            t_name = str(row[text_col])                # GPT text embeddings keyed by raw name
            i_name = str(row[image_col]) + image_suffix  # glyph embeddings keyed by name + ".png"

            i_text = text_name_to_idx.get(t_name, None)
            i_img = img_name_to_idx.get(i_name, None)

            if i_text is None:
                missing_text += 1
                continue
            if i_img is None:
                missing_img += 1
                continue

            lbl_raw = int(row[label_col])
            if lbl_raw == 1:
                lbl = 1.0
            elif lbl_raw == 0:
                lbl = 0.0
            else:
                bad_label += 1
                continue

            text_indices.append(i_text)
            img_indices.append(i_img)
            labels.append(lbl)

        kept = len(labels)
        skipped = total_pairs - kept

        print(
            f"{split_name}: loaded {total_pairs} pairs, "
            f"kept {kept}, skipped {skipped} "
            f"(missing text: {missing_text}, missing image: {missing_img}, bad labels: {bad_label})."
        )

        if kept == 0:
            raise ValueError(f"No valid pairs found for split {split_name}.")

        self.text_idx = torch.as_tensor(np.array(text_indices, dtype=np.int64), dtype=torch.long)
        self.img_idx = torch.as_tensor(np.array(img_indices, dtype=np.int64), dtype=torch.long)
        self.labels = torch.as_tensor(np.array(labels, dtype=np.float32), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.text_emb[self.text_idx[idx]],
            self.img_emb[self.img_idx[idx]],
            self.labels[idx],
        )


# -----------------------------
# Model & loss
# -----------------------------


class TextToImageMLP(nn.Module):
    """
    2–3 layer MLP mapping text embeddings -> image embedding space.
    """

    def __init__(self, text_dim, img_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, img_dim),
        )

    def forward(self, x):
        x = self.net(x)
        # Normalize so we can compare with normalized image embeddings
        return nn.functional.normalize(x, p=2, dim=1)


def contrastive_loss(z1, z2, y, margin):
    """
    Contrastive loss with margin, using Euclidean distance.

    z1, z2 : (B, D)
    y      : (B,)  (1 for positive, 0 for negative)
    margin : float
    """
    distances = torch.norm(z1 - z2, p=2, dim=1)
    pos_loss = y * distances.pow(2)
    neg_loss = (1.0 - y) * torch.clamp(margin - distances, min=0.0).pow(2)
    loss = (pos_loss + neg_loss).mean()
    return loss, distances


# -----------------------------
# Training / evaluation
# -----------------------------


def get_stage(epoch, num_epochs):
    """Return curriculum stage index 1, 2, or 3."""
    third = num_epochs / 3.0
    if epoch < third:
        return 1
    elif epoch < 2 * third:
        return 2
    else:
        return 3


def make_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


def evaluate_model(model, dataset, device, margin, roc_plot_path=None, split_name="val"):
    """
    Run full evaluation on a dataset.

    Returns
    -------
    metrics : dict
        Includes loss, roc_auc, accuracy, threshold, tn, fp, fn, tp.
    """
    model.eval()
    loader = make_dataloader(dataset, batch_size=512, shuffle=False)

    all_labels = []
    all_scores = []  # similarity scores = -distance
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for text_emb, img_emb, y in loader:
            text_emb = text_emb.to(device)
            img_emb = img_emb.to(device)
            y = y.to(device)

            z_text = model(text_emb)
            z_img = nn.functional.normalize(img_emb, p=2, dim=1)

            loss, distances = contrastive_loss(z_text, z_img, y, margin)
            scores = -distances  # higher = more similar / more likely positive

            total_loss += loss.item()
            total_batches += 1

            all_labels.extend(y.cpu().numpy().tolist())
            all_scores.extend(scores.cpu().numpy().tolist())

    eval_loss = total_loss / max(total_batches, 1)

    y_true = np.array(all_labels, dtype=np.float32)
    scores = np.array(all_scores, dtype=np.float32)

    unique_labels = np.unique(y_true)
    if unique_labels.size < 2:
        roc_auc = float("nan")
        y_pred = np.full_like(y_true, fill_value=int(unique_labels[0]))
        acc = accuracy_score(y_true, y_pred)
        tn = fp = fn = tp = 0
        if unique_labels[0] == 1:
            tp = int((y_true == 1).sum())
        else:
            tn = int((y_true == 0).sum())

        metrics = {
            "loss": float(eval_loss),
            "roc_auc": roc_auc,
            "acc": float(acc),
            "threshold": None,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
        return metrics

    # Normal case
    roc_auc = roc_auc_score(y_true, scores)
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = thresholds[best_idx]

    y_pred = (scores >= best_threshold).astype(np.int32)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Optional ROC plot
    if roc_plot_path is not None:
        plt.figure()
        plt.plot(fpr, tpr, label=f"{split_name} ROC (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve ({split_name})")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_plot_path, dpi=150)
        plt.close()
        print(f"Saved ROC curve to {roc_plot_path}")

    metrics = {
        "loss": float(eval_loss),
        "roc_auc": float(roc_auc),
        "acc": float(acc),
        "threshold": float(best_threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def run_training(
    train_easy,
    train_medium,
    train_hard,
    val_dataset,
    text_dim,
    img_dim,
    device,
    epochs=10,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    hidden_dim=512,
    margin=1.0,
    output_dir=None,
    log_to_stdout=True,
    make_plots=True,
    save_best=True,
):
    batch_size = min(batch_size, 512)

    model = TextToImageMLP(text_dim=text_dim, img_dim=img_dim, hidden_dim=hidden_dim)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_roc_auc": [],
    }

    best_state_dict = None
    best_val_roc_auc = -float("inf")
    best_epoch_metrics = None

    num_workers = 0

    if log_to_stdout:
        print(
            f"Starting training for {epochs} epochs | "
            f"batch_size={batch_size}, lr={lr:.3e}, margin={margin}"
        )

    current_stage = None
    train_loader = None

    for epoch in range(epochs):
        stage = get_stage(epoch, epochs)
        if stage != current_stage:
            if stage == 1:
                train_dataset = train_easy
            elif stage == 2:
                train_dataset = ConcatDataset([train_easy, train_medium])
            else:
                train_dataset = ConcatDataset([train_easy, train_medium, train_hard])

            train_loader = make_dataloader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
            current_stage = stage

            if log_to_stdout:
                print(f"\n>>> Epoch {epoch+1}/{epochs}: switching to stage {stage}")

        # -------- Training epoch --------
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        if log_to_stdout:
            progress_bar = tqdm(train_loader, desc=f"Train epoch {epoch+1}/{epochs}")
        else:
            progress_bar = train_loader

        distance_threshold = margin / 2.0

        for text_emb, img_emb, y in progress_bar:
            text_emb = text_emb.to(device)
            img_emb = img_emb.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            z_text = model(text_emb)
            z_img = nn.functional.normalize(img_emb, p=2, dim=1)

            loss, distances = contrastive_loss(z_text, z_img, y, margin)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * text_emb.size(0)

            preds = (distances <= distance_threshold).float()
            running_correct += (preds == y).sum().item()
            running_total += text_emb.size(0)

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        # -------- Validation --------
        val_metrics = evaluate_model(
            model,
            val_dataset,
            device,
            margin,
            roc_plot_path=os.path.join(output_dir, f"roc_val_epoch{epoch+1}.png")
            if (make_plots and output_dir is not None)
            else None,
            split_name=f"val_epoch{epoch+1}",
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])

        if log_to_stdout:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"val_acc={val_metrics['acc']:.4f}, "
                f"val_roc_auc={val_metrics['roc_auc']:.4f}"
            )

        val_auc = val_metrics["roc_auc"]
        if (val_auc is not None) and (not np.isnan(val_auc)) and (val_auc > best_val_roc_auc):
            best_val_roc_auc = val_auc
            best_state_dict = deepcopy(model.state_dict())
            best_epoch_metrics = val_metrics

            if log_to_stdout:
                print(
                    f"*** New best model at epoch {epoch+1} "
                    f"(val_roc_auc={best_val_roc_auc:.4f})"
                )

    # ----------------- End of training -----------------
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Save best weights and metrics if requested
    if save_best and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

        if best_state_dict is not None and best_epoch_metrics is not None:
            weights_path = os.path.join(output_dir, "text2image_mlp_best.pt")
            torch.save(best_state_dict, weights_path)

            metrics_path = os.path.join(output_dir, "validation_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "best_val_roc_auc": best_val_roc_auc,
                        "best_val_acc": best_epoch_metrics["acc"],
                        "best_val_loss": best_epoch_metrics["loss"],
                        "threshold": best_epoch_metrics["threshold"],
                        "tn": best_epoch_metrics["tn"],
                        "fp": best_epoch_metrics["fp"],
                        "fn": best_epoch_metrics["fn"],
                        "tp": best_epoch_metrics["tp"],
                        "history": history,
                        "hyperparams": {
                            "epochs": epochs,
                            "batch_size": batch_size,
                            "lr": lr,
                            "weight_decay": weight_decay,
                            "hidden_dim": hidden_dim,
                            "margin": margin,
                            "text_dim": text_dim,
                            "img_dim": img_dim,
                        },
                    },
                    f,
                    indent=2,
                )
        else:
            if log_to_stdout:
                print(
                    "Warning: no valid ROC AUC across epochs "
                    "(single-class validation set?). "
                    "Not saving best weights/metrics."
                )
            fallback_path = os.path.join(output_dir, "training_history_only.json")
            with open(fallback_path, "w") as f:
                json.dump(
                    {
                        "history": history,
                        "hyperparams": {
                            "epochs": epochs,
                            "batch_size": batch_size,
                            "lr": lr,
                            "weight_decay": weight_decay,
                            "hidden_dim": hidden_dim,
                            "margin": margin,
                            "text_dim": text_dim,
                            "img_dim": img_dim,
                        },
                    },
                    f,
                    indent=2,
                )

    # Plot training curves
    if make_plots and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        epochs_range = list(range(1, epochs + 1))

        # Accuracy plot
        plt.figure()
        plt.plot(epochs_range, history["train_acc"], label="Train accuracy")
        plt.plot(epochs_range, history["val_acc"], label="Val accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "accuracy_curves.png"))
        plt.close()

        # Loss plot
        plt.figure()
        plt.plot(epochs_range, history["train_loss"], label="Train loss")
        plt.plot(epochs_range, history["val_loss"], label="Val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "loss_curves.png"))
        plt.close()

    return {
        "best_val_roc_auc": best_val_roc_auc
        if (best_epoch_metrics is not None)
        else None,
        "best_epoch_metrics": best_epoch_metrics,
        "history": history,
        "model": model,
    }


# -----------------------------
# CLI
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 2–3 layer MLP mapping GPT text embeddings to image embeddings."
    )

    # Embeddings
    parser.add_argument("--train-text-embeddings", type=str, required=True)
    parser.add_argument("--val-text-embeddings", type=str, required=True)
    parser.add_argument("--test-text-embeddings", type=str, required=False)

    parser.add_argument("--train-image-embeddings", type=str, required=True)
    parser.add_argument("--val-image-embeddings", type=str, required=True)
    parser.add_argument("--test-image-embeddings", type=str, required=False)

    # Pair files (easy/med/hard for train, one for val/test)
    parser.add_argument("--easy-pairs", type=str, required=True)
    parser.add_argument("--medium-pairs", type=str, required=True)
    parser.add_argument("--hard-pairs", type=str, required=True)
    parser.add_argument("--val-pairs", type=str, required=True)
    parser.add_argument("--test-pairs", type=str, required=False)

    # Column names
    parser.add_argument("--text-col", type=str, default="fraudulent_name")
    parser.add_argument("--image-col", type=str, default="real_name")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--image-suffix", type=str, default=".png")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--margin", type=float, default=1.0)

    # Misc
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/text2image_mlp",
        help="Directory to save weights, plots, and metrics.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-logging", action="store_true")

    # Optuna
    parser.add_argument("--optuna-trials", type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load embeddings
    print("Loading train text embeddings...")
    train_text_np, train_text_name_to_idx = load_embeddings_npz(args.train_text_embeddings)
    print("Loading val text embeddings...")
    val_text_np, val_text_name_to_idx = load_embeddings_npz(args.val_text_embeddings)

    print("Loading train image embeddings...")
    train_img_np, train_img_name_to_idx = load_embeddings_npz(args.train_image_embeddings)
    print("Loading val image embeddings...")
    val_img_np, val_img_name_to_idx = load_embeddings_npz(args.val_image_embeddings)

    text_dim = train_text_np.shape[1]
    img_dim = train_img_np.shape[1]
    print(f"text_dim={text_dim}, image_dim={img_dim}")

    train_text_tensor = torch.from_numpy(train_text_np).float()
    val_text_tensor = torch.from_numpy(val_text_np).float()
    train_img_tensor = torch.from_numpy(train_img_np).float()
    val_img_tensor = torch.from_numpy(val_img_np).float()

    # Build datasets
    print("\nBuilding training datasets (easy/medium/hard)...")
    train_easy = TextImagePairDataset(
        parquet_path=args.easy_pairs,
        text_emb_tensor=train_text_tensor,
        img_emb_tensor=train_img_tensor,
        text_name_to_idx=train_text_name_to_idx,
        img_name_to_idx=train_img_name_to_idx,
        split_name="train_easy",
        text_col=args.text_col,
        image_col=args.image_col,
        label_col=args.label_col,
        image_suffix=args.image_suffix,
    )
    train_medium = TextImagePairDataset(
        parquet_path=args.medium_pairs,
        text_emb_tensor=train_text_tensor,
        img_emb_tensor=train_img_tensor,
        text_name_to_idx=train_text_name_to_idx,
        img_name_to_idx=train_img_name_to_idx,
        split_name="train_medium",
        text_col=args.text_col,
        image_col=args.image_col,
        label_col=args.label_col,
        image_suffix=args.image_suffix,
    )
    train_hard = TextImagePairDataset(
        parquet_path=args.hard_pairs,
        text_emb_tensor=train_text_tensor,
        img_emb_tensor=train_img_tensor,
        text_name_to_idx=train_text_name_to_idx,
        img_name_to_idx=train_img_name_to_idx,
        split_name="train_hard",
        text_col=args.text_col,
        image_col=args.image_col,
        label_col=args.label_col,
        image_suffix=args.image_suffix,
    )

    print("\nBuilding validation dataset...")
    val_dataset = TextImagePairDataset(
        parquet_path=args.val_pairs,
        text_emb_tensor=val_text_tensor,
        img_emb_tensor=val_img_tensor,
        text_name_to_idx=val_text_name_to_idx,
        img_name_to_idx=val_img_name_to_idx,
        split_name="validate",
        text_col=args.text_col,
        image_col=args.image_col,
        label_col=args.label_col,
        image_suffix=args.image_suffix,
    )

    # (optional) load test embeddings/dataset
    test_dataset = None
    if args.test_text_embeddings and args.test_image_embeddings and args.test_pairs:
        print("\nLoading test embeddings...")
        test_text_np, test_text_name_to_idx = load_embeddings_npz(args.test_text_embeddings)
        test_img_np, test_img_name_to_idx = load_embeddings_npz(args.test_image_embeddings)
        test_text_tensor = torch.from_numpy(test_text_np).float()
        test_img_tensor = torch.from_numpy(test_img_np).float()

        print("Building test dataset...")
        test_dataset = TextImagePairDataset(
            parquet_path=args.test_pairs,
            text_emb_tensor=test_text_tensor,
            img_emb_tensor=test_img_tensor,
            text_name_to_idx=test_text_name_to_idx,
            img_name_to_idx=test_img_name_to_idx,
            split_name="test",
            text_col=args.text_col,
            image_col=args.image_col,
            label_col=args.label_col,
            image_suffix=args.image_suffix,
        )

    # TODO: you can plug Optuna here similar to your existing script if you want.
    # For now, we skip Optuna for clarity.

    print("\nStarting final training run with hyperparameters:")
    print(
        f"  epochs={args.epochs}, batch_size={args.batch_size}, "
        f"lr={args.lr}, weight_decay={args.weight_decay}, "
        f"hidden_dim={args.hidden_dim}, margin={args.margin}"
    )

    result = run_training(
        train_easy=train_easy,
        train_medium=train_medium,
        train_hard=train_hard,
        val_dataset=val_dataset,
        text_dim=text_dim,
        img_dim=img_dim,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        margin=args.margin,
        output_dir=args.output_dir,
        log_to_stdout=not args.no_logging,
        make_plots=not args.no_plots,
        save_best=True,
    )

    best = result["best_epoch_metrics"]
    model = result["model"]

    if best is None:
        history = result["history"]
        last_epoch = len(history["val_loss"])
        print(
            "\n=== No valid 'best' epoch (ROC AUC undefined, likely single-class validation set). ==="
        )
        if last_epoch > 0:
            print(
                f"Showing last-epoch metrics instead (epoch {last_epoch}, no threshold/CM):"
            )
            print(f"Val loss:   {history['val_loss'][-1]:.4f}")
            print(f"Val acc:    {history['val_acc'][-1]:.4f}")
            print(f"Train loss: {history['train_loss'][-1]:.4f}")
            print(f"Train acc:  {history['train_acc'][-1]:.4f}")
        else:
            print("No epochs recorded in history.")
        return

    print("\n=== Final validation metrics (best epoch) ===")
    print(f"ROC AUC:   {best['roc_auc']:.4f}")
    print(f"Accuracy:  {best['acc']:.4f}")
    print(f"Loss:      {best['loss']:.4f}")
    print(f"Threshold (Youden): {best['threshold']:.6f}")
    print(f"TN: {best['tn']}  FP: {best['fp']}  FN: {best['fn']}  TP: {best['tp']}")

    # Evaluate on test set (if available)
    if test_dataset is not None:
        print("\nEvaluating on test set...")
        test_metrics = evaluate_model(
            model,
            test_dataset,
            device,
            args.margin,
            roc_plot_path=os.path.join(args.output_dir, "roc_test.png")
            if not args.no_plots
            else None,
            split_name="test",
        )
        print("\n=== Test metrics ===")
        print(f"ROC AUC:   {test_metrics['roc_auc']:.4f}")
        print(f"Accuracy:  {test_metrics['acc']:.4f}")
        print(f"Loss:      {test_metrics['loss']:.4f}")
        print(f"Threshold (Youden): {test_metrics['threshold']}")
        print(
            f"TN: {test_metrics['tn']}  FP: {test_metrics['fp']}  "
            f"FN: {test_metrics['fn']}  TP: {test_metrics['tp']}"
        )


if __name__ == "__main__":
    main()


# command line

# python3 text_to_image_emb/train_mlp.py \
#   --train-text-embeddings data/embeddings/text_embeddings/train_gpt_embs.npz \
#   --val-text-embeddings   data/embeddings/text_embeddings/val_gpt_embs.npz \
#   --test-text-embeddings  data/embeddings/text_embeddings/test_gpt_embs.npz \
#   --train-image-embeddings data/embeddings/siglip_glyphs/image_embeddings_train.npz \
#   --val-image-embeddings   data/embeddings/siglip_glyphs/image_embeddings_validate.npz \
#   --test-image-embeddings  data/embeddings/siglip_glyphs/image_embeddings_test.npz \
#   --easy-pairs   data/processed/train_pairs_easy_100k.parquet \
#   --medium-pairs data/processed/train_pairs_medium_100k.parquet \
#   --hard-pairs   data/processed/train_pairs_hard_100k.parquet \
#   --val-pairs    data/processed/validate_pairs_ref_10k.parquet \
#   --test-pairs   data/processed/test_pairs_ref_10k.parquet \
#   --epochs 10 \
#   --batch-size 512 \
#   --lr 1e-3 \
#   --weight-decay 1e-4 \
#   --hidden-dim 512 \
#   --margin 1.0
