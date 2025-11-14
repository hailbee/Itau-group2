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



class PairDataset(Dataset):
    """
    Dataset of (fraudulent, real, label) pairs backed by a shared embedding tensor.
    """

    def __init__(self, emb_tensor, left_indices, right_indices, labels):
        """
        Parameters
        ----------
        emb_tensor : torch.FloatTensor, shape (N, D)
            All embeddings for this split (on CPU).
        left_indices, right_indices : 1D np.ndarray[int]
            Indices into emb_tensor.
        labels : 1D np.ndarray[float]
            Labels (0.0 for negative pair, 1.0 for positive pair).
        """
        self.emb = emb_tensor
        self.left = torch.as_tensor(left_indices, dtype=torch.long)
        self.right = torch.as_tensor(right_indices, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.emb[self.left[idx]],
            self.emb[self.right[idx]],
            self.labels[idx],
        )


def build_pair_dataset(parquet_path, emb_tensor, name_to_idx, split_name):
    """
    Create a PairDataset from a parquet file and embedding lookup.

    Parameters
    ----------
    parquet_path : str
    emb_tensor : torch.FloatTensor
    name_to_idx : dict[str, int]
    split_name : str
        For logging only.

    Returns
    -------
    dataset : PairDataset
    """
    df = pd.read_parquet(parquet_path)
    required_cols = {"fraudulent_name", "real_name", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{parquet_path} missing columns: {missing}")

    total_pairs = len(df)
    left_indices = []
    right_indices = []
    labels = []

    for _, row in df.iterrows():
        fraud_name = str(row["fraudulent_name"]) + ".png"
        real_name = str(row["real_name"]) + ".png"

        i_left = name_to_idx.get(fraud_name, None)
        i_right = name_to_idx.get(real_name, None)

        if i_left is None or i_right is None:
            continue

        # Map labels: 1 -> positive, 2 -> negative
        lbl_raw = int(row["label"])
        if lbl_raw == 1:
            lbl = 1.0
        elif lbl_raw == 2:
            lbl = 0.0
        else:
            # Skip unexpected label values
            continue

        left_indices.append(i_left)
        right_indices.append(i_right)
        labels.append(lbl)

    kept = len(labels)
    skipped = total_pairs - kept
    print(
        f"{split_name}: loaded {total_pairs} pairs, "
        f"kept {kept} with embeddings, skipped {skipped}."
    )

    if kept == 0:
        raise ValueError(f"No valid pairs found for split {split_name}.")

    return PairDataset(
        emb_tensor=emb_tensor,
        left_indices=np.array(left_indices, dtype=np.int64),
        right_indices=np.array(right_indices, dtype=np.int64),
        labels=np.array(labels, dtype=np.float32),
    )


# -----------------------------
# Model & loss
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


def contrastive_loss(z1, z2, y, margin):
    """
    Contrastive loss with margin, using Euclidean distance.

    Parameters
    ----------
    z1, z2 : torch.FloatTensor, shape (B, D)
    y : torch.FloatTensor, shape (B,)  (1 for positive, 0 for negative)
    margin : float

    Returns
    -------
    loss : torch.FloatTensor (scalar)
    distances : torch.FloatTensor, shape (B,)
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
        pin_memory=True,
    )


def evaluate_model(model, val_dataset, device, margin):
    """
    Run full evaluation on validation dataset.

    Returns
    -------
    metrics : dict
        Includes val_loss, roc_auc, accuracy, threshold, tn, fp, fn, tp.
    """
    model.eval()
    val_loader = make_dataloader(val_dataset, batch_size=512, shuffle=False)

    all_labels = []
    all_scores = []  # similarity scores = -distance
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for x1, x2, y in val_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            z1 = model(x1)
            z2 = model(x2)
            z1 = nn.functional.normalize(z1, p=2, dim=1)
            z2 = nn.functional.normalize(z2, p=2, dim=1)

            loss, distances = contrastive_loss(z1, z2, y, margin)
            scores = -distances  # higher is more similar / more likely positive

            total_loss += loss.item()
            total_batches += 1

            all_labels.extend(y.cpu().numpy().tolist())
            all_scores.extend(scores.cpu().numpy().tolist())

    val_loss = total_loss / max(total_batches, 1)

    y_true = np.array(all_labels, dtype=np.float32)
    scores = np.array(all_scores, dtype=np.float32)

    # --- Handle single-class edge case ---
    unique_labels = np.unique(y_true)
    if unique_labels.size < 2:
        # Can't compute ROC AUC or meaningful Youden threshold
        roc_auc = float("nan")
        # Degenerate classification: predict everything as the only class
        y_pred = np.full_like(y_true, fill_value=int(unique_labels[0]))
        acc = accuracy_score(y_true, y_pred)
        tn = fp = fn = tp = 0
        # simple confusion matrix for single-class case
        if unique_labels[0] == 1:
            tp = int((y_true == 1).sum())
        else:
            tn = int((y_true == 0).sum())

        return {
            "val_loss": float(val_loss),
            "val_roc_auc": roc_auc,
            "val_acc": float(acc),
            "threshold": None,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

    # --- Normal case (both classes present) ---
    roc_auc = roc_auc_score(y_true, scores)
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    best_threshold = thresholds[best_idx]

    y_pred = (scores >= best_threshold).astype(np.int32)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "val_loss": float(val_loss),
        "val_roc_auc": float(roc_auc),
        "val_acc": float(acc),
        "threshold": float(best_threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_training(
    train_easy,
    train_medium,
    train_hard,
    val_dataset,
    input_dim,
    device,
    epochs=10,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    hidden_dim=512,
    output_dim=256,
    margin=1.0,
    output_dir=None,
    log_to_stdout=True,
    make_plots=True,
    save_best=True,
):
    """
    Train the MLP head with curriculum learning and return metrics.

    For Optuna trials, set:
        make_plots=False, save_best=False, output_dir=None, maybe fewer epochs.
    """
    batch_size = min(batch_size, 512)

    model = MLPHead(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
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

    num_workers = 0  # safe default for most laptops/OSes

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
            # Rebuild train loader according to curriculum stage
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

        # Use a simple distance threshold for approximate training accuracy
        distance_threshold = margin / 2.0

        for x1, x2, y in progress_bar:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            z1 = model(x1)
            z2 = model(x2)
            z1 = nn.functional.normalize(z1, p=2, dim=1)
            z2 = nn.functional.normalize(z2, p=2, dim=1)

            loss, distances = contrastive_loss(z1, z2, y, margin)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x1.size(0)

            # approximate train accuracy using distance threshold
            preds = (distances <= distance_threshold).float()
            running_correct += (preds == y).sum().item()
            running_total += x1.size(0)

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        # -------- Validation --------
        val_metrics = evaluate_model(model, val_dataset, device, margin)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["val_loss"])
        history["val_acc"].append(val_metrics["val_acc"])
        history["val_roc_auc"].append(val_metrics["val_roc_auc"])

        if log_to_stdout:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f}, "
                f"val_acc={val_metrics['val_acc']:.4f}, "
                f"val_roc_auc={val_metrics['val_roc_auc']:.4f}"
            )

        # Track best model by validation ROC AUC
        # if val_metrics["val_roc_auc"] > best_val_roc_auc:
        #     best_val_roc_auc = val_metrics["val_roc_auc"]
        #     best_state_dict = deepcopy(model.state_dict())
        #     best_epoch_metrics = val_metrics

        #     if log_to_stdout:
        #         print(
        #             f"*** New best model at epoch {epoch+1} "
        #             f"(val_roc_auc={best_val_roc_auc:.4f})"
        #         )

        # Track best model by validation ROC AUC (ignore NaNs)
        val_auc = val_metrics["val_roc_auc"]

        # np.isnan handles the single-class case where we set roc_auc = nan
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
            # Normal case: we have a valid best epoch
            weights_path = os.path.join(output_dir, "siglip_mlp_head_best.pt")
            torch.save(best_state_dict, weights_path)

            metrics_path = os.path.join(output_dir, "validation_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(
                    {
                        "best_val_roc_auc": best_val_roc_auc,
                        "best_val_acc": best_epoch_metrics["val_acc"],
                        "best_val_loss": best_epoch_metrics["val_loss"],
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
                            "output_dim": output_dim,
                            "margin": margin,
                        },
                    },
                    f,
                    indent=2,
                )
        else:
            # Edge case: no valid ROC AUC (e.g. single-class validation for all epochs)
            # We can still save history + hyperparams for debugging.
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
                            "output_dim": output_dim,
                            "margin": margin,
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

    # 👇 MAKE SURE THIS IS HERE AND INDENTED AT THE SAME LEVEL AS THE 'if make_plots...' BLOCK
    return {
        "best_val_roc_auc": best_val_roc_auc
        if (best_epoch_metrics is not None)
        else None,
        "best_epoch_metrics": best_epoch_metrics,
        "history": history,
    }



# -----------------------------
# Main / Optuna
# -----------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 2-layer MLP head on SigLIP glyph embeddings "
        "using contrastive loss with curriculum learning."
    )

    # Embeddings
    parser.add_argument(
        "--train-embeddings",
        type=str,
        required=True,
        help="Path to train image embeddings .npz",
    )
    parser.add_argument(
        "--val-embeddings",
        type=str,
        required=True,
        help="Path to validation image embeddings .npz",
    )

    # Pair files
    parser.add_argument(
        "--easy-pairs",
        type=str,
        required=True,
        help="Path to train_pairs_easy parquet file",
    )
    parser.add_argument(
        "--medium-pairs",
        type=str,
        required=True,
        help="Path to train_pairs_medium parquet file",
    )
    parser.add_argument(
        "--hard-pairs",
        type=str,
        required=True,
        help="Path to train_pairs_hard parquet file",
    )
    parser.add_argument(
        "--val-pairs",
        type=str,
        required=True,
        help="Path to validation pairs parquet file",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--margin", type=float, default=1.0)

    # Misc
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'cuda', 'cpu', or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ft_image_encoder",
        help="Directory to save weights, plots, and metrics.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable saving training curves.",
    )
    parser.add_argument(
        "--no-logging",
        action="store_true",
        help="Disable stdout logging (except Optuna summary).",
    )

    # Optuna
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=0,
        help="Number of Optuna trials for hyperparameter search. "
        "If 0, Optuna is not used.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load embeddings
    print("Loading train embeddings...")
    train_emb_np, train_name_to_idx = load_embeddings_npz(args.train_embeddings, max_items=None)

    print("Loading validation embeddings...")
    val_emb_np, val_name_to_idx = load_embeddings_npz(args.val_embeddings, max_items=None)

    input_dim = train_emb_np.shape[1]
    print(f"Embedding dimension: {input_dim}")

    # Convert to torch tensors (CPU)
    train_emb_tensor = torch.from_numpy(train_emb_np).float()
    val_emb_tensor = torch.from_numpy(val_emb_np).float()

    # Build datasets
    print("\nBuilding training datasets...")
    train_easy = build_pair_dataset(
        args.easy_pairs, train_emb_tensor, train_name_to_idx, "train_easy"
    )
    train_medium = build_pair_dataset(
        args.medium_pairs, train_emb_tensor, train_name_to_idx, "train_medium"
    )
    train_hard = build_pair_dataset(
        args.hard_pairs, train_emb_tensor, train_name_to_idx, "train_hard"
    )

    print("\nBuilding validation dataset...")
    val_dataset = build_pair_dataset(
        args.val_pairs, val_emb_tensor, val_name_to_idx, "validate"
    )

    # Optuna hyperparameter search (optional)
    if args.optuna_trials and args.optuna_trials > 0:
        print(f"\nRunning Optuna hyperparameter search ({args.optuna_trials} trials)...")

        def objective(trial):
            margin = trial.suggest_float("margin", 0.5, 2.0)
            lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
            hidden_dim = trial.suggest_categorical(
                "hidden_dim", [256, 512, 768]
            )
            output_dim = trial.suggest_categorical(
                "output_dim", [128, 256, 384]
            )
            batch_size = trial.suggest_categorical(
                "batch_size", [128, 256, 512]
            )

            # Use a small number of epochs per trial
            trial_epochs = min(3, args.epochs)

            result = run_training(
                train_easy=train_easy,
                train_medium=train_medium,
                train_hard=train_hard,
                val_dataset=val_dataset,
                input_dim=input_dim,
                device=device,
                epochs=trial_epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=args.weight_decay,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                margin=margin,
                output_dir=None,
                log_to_stdout=False,
                make_plots=False,
                save_best=False,
            )

            return result["best_val_roc_auc"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.optuna_trials)

        print("\nOptuna finished.")
        print(f"Best value (val ROC AUC): {study.best_value:.4f}")
        print("Best hyperparameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        # Override args with best hyperparams
        args.margin = float(study.best_params.get("margin", args.margin))
        args.lr = float(study.best_params.get("lr", args.lr))
        args.hidden_dim = int(study.best_params.get("hidden_dim", args.hidden_dim))
        args.output_dim = int(study.best_params.get("output_dim", args.output_dim))
        args.batch_size = int(study.best_params.get("batch_size", args.batch_size))

    # Final full training run with (possibly tuned) hyperparameters
    print("\nStarting final training run with hyperparameters:")
    print(
        f"  epochs={args.epochs}, batch_size={args.batch_size}, "
        f"lr={args.lr}, weight_decay={args.weight_decay}, "
        f"hidden_dim={args.hidden_dim}, output_dim={args.output_dim}, "
        f"margin={args.margin}"
    )

    result = run_training(
        train_easy=train_easy,
        train_medium=train_medium,
        train_hard=train_hard,
        val_dataset=val_dataset,
        input_dim=input_dim,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        margin=args.margin,
        output_dir=args.output_dir,
        log_to_stdout=not args.no_logging,
        make_plots=not args.no_plots,
        save_best=True,
    )

    best = result["best_epoch_metrics"]

    if best is None:
        # This happens when validation ROC AUC was never valid (e.g. single-class validation set)
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
    print(f"ROC AUC:   {best['val_roc_auc']:.4f}")
    print(f"Accuracy:  {best['val_acc']:.4f}")
    print(f"Loss:      {best['val_loss']:.4f}")
    print(f"Threshold (Youden): {best['threshold']:.6f}")
    print(f"TN: {best['tn']}  FP: {best['fp']}  FN: {best['fn']}  TP: {best['tp']}")



if __name__ == "__main__":
    main()


# run this

# python ft_image_encoder/train_head.py \
#   --train-embeddings data/embeddings/siglip_glyphs/image_embeddings_train.npz \
#   --val-embeddings   data/embeddings/siglip_glyphs/image_embeddings_validate.npz \
#   --easy-pairs   data/processed/train_pairs_easy_100k.parquet \
#   --medium-pairs data/processed/train_pairs_medium_100k.parquet \
#   --hard-pairs   data/processed/train_pairs_hard_100k.parquet \
#   --val-pairs    data/processed/validate_pairs_ref_10k.parquet \
#   --epochs 10 \
#   --batch-size 512 \
#   --lr 1e-3 \
#   --weight-decay 1e-4 \
#   --hidden-dim 512 \
#   --output-dim 256 \
#   --margin 1.0 \
#   --optuna-trials 10
