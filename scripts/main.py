# scripts/main.py
from __future__ import annotations

# --- repo-root path bootstrap ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo root (one level up from scripts/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# --- end bootstrap ---

import argparse
import os
from typing import Dict, List, Tuple
from tqdm import tqdm

from scripts.optimization.unified_optimizer import UnifiedHyperparameterOptimizer

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from model_utils.models.model_factory import ModelFactory
from model_utils.models.base import BaseSiameseModel

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt



# ------------------------------
# Data loading / metrics helpers
# ------------------------------

def load_pairs_dataframe(path: str, left_col: str, right_col: str, label_col: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[-1].lower()
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported pairs file type: {ext}. Use .csv, .tsv, or .parquet")

    missing = [c for c in (left_col, right_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Pairs file missing required columns: {missing}. Found: {list(df.columns)}")

    return df[[left_col, right_col, label_col]].copy()


@torch.inference_mode()
def compute_pairwise_scores(
    model: BaseSiameseModel,
    left: List[str],
    right: List[str],
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      scores: [N] float32 (NaN where invalid/missing)
      valid:  [N] bool mask
    """
    assert len(left) == len(right)
    N = len(left)
    scores = np.full(N, np.nan, dtype=np.float32)
    valid = np.zeros(N, dtype=bool)

    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        L = left[i:j]
        R = right[i:j]

        # We call model.encode() twice with aligned inputs
        ZL = model.encode(L)  # [B, D]
        ZR = model.encode(R)  # [B, D]

        # If using the precomputed backbone in non-strict mode, missing words are zero vectors.
        # Detect valid rows via norm>0 on both sides (pre-normalized -> norm==0 only for zeros).
        L_norm2 = (ZL ** 2).sum(dim=1).cpu().numpy()
        R_norm2 = (ZR ** 2).sum(dim=1).cpu().numpy()
        ok = (L_norm2 > 1e-12) & (R_norm2 > 1e-12)

        if np.any(ok):
            sims = (ZL[ok] * ZR[ok]).sum(dim=1).cpu().numpy()  # cosine == dot product after normalization
            idxs = np.where(ok)[0]
            for u, s in zip(idxs, sims):
                scores[i + int(u)] = float(s)
                valid[i + int(u)] = True

    return scores, valid

@torch.inference_mode()
def compute_pairwise_embeddings(
    model: BaseSiameseModel,
    left: List[str],
    right: List[str],
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      eL:   (N, D) float32 (NaN rows where invalid)
      eR:   (N, D) float32 (NaN rows where invalid)
      valid:(N,)  bool
    """
    assert len(left) == len(right)
    N = len(left)
    # probe a batch to get D
    ZL0 = model.encode(left[:1])
    D = int(ZL0.shape[1])
    eL = np.full((N, D), np.nan, dtype=np.float32)
    eR = np.full((N, D), np.nan, dtype=np.float32)
    valid = np.zeros(N, dtype=bool)

    for i in tqdm(range(0, N, batch_size)):
        j = min(i + batch_size, N)
        L = left[i:j]; R = right[i:j]
        ZL = model.encode(L)  # [B, D]
        ZR = model.encode(R)  # [B, D]
        L_norm2 = (ZL ** 2).sum(dim=1).cpu().numpy()
        R_norm2 = (ZR ** 2).sum(dim=1).cpu().numpy()
        ok = (L_norm2 > 1e-12) & (R_norm2 > 1e-12)
        if np.any(ok):
            idxs = np.where(ok)[0]
            eL[i + idxs] = ZL[ok].cpu().numpy().astype(np.float32)
            eR[i + idxs] = ZR[ok].cpu().numpy().astype(np.float32)
            valid[i + idxs] = True
    return eL, eR, valid

def pick_thresholds(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    fpr, tpr, thr = roc_curve(labels, scores)
    youden_idx = int(np.argmax(tpr - fpr))
    thr_youden = float(thr[youden_idx])

    # Max-accuracy across same threshold grid
    preds_grid = (scores[:, None] >= thr[None, :])
    accs = (preds_grid == labels[:, None]).mean(axis=0)
    thr_maxacc = float(thr[int(np.argmax(accs))])

    return {"youden": thr_youden, "max_acc": thr_maxacc}


def metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, thr: float) -> Dict[str, float]:
    preds = (scores >= thr).astype(int)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return {
        "threshold": float(thr),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

def run_pairwise_mlp(args: argparse.Namespace) -> None:
    """
    Train/evaluate the pairwise-MLP head on top of the backbone embeddings.
    Uses args.mlp_split (default 0.40) for training; the remainder is eval.
    """
    from model_utils.features.pairwise import pairwise_features
    from model_utils.trainers.pairwise_mlp_trainer import train_pairwise_mlp, best_youden_threshold
    from model_utils.heads.pairwise_mlp import PairwiseMLP

    # 1) Build embedding model (same as baseline)
    if args.backbone.lower() == "precomputed":
        wrapper = ModelFactory.create_model(
            "precomputed",
            npz_paths=args.npz,
            lowercase_keys=not args.pc_case_sensitive,
            normalize=args.pc_norm,
            strict=not args.pc_non_strict,
        )
        model = BaseSiameseModel(backbone=wrapper, projection_dim=wrapper.embedding_dim, use_projector=False)
        print(f"[pairwise-mlp] using precomputed backbone with embedding_dim={wrapper.embedding_dim}")
    else:
        wrapper = ModelFactory.create_model(args.backbone, device=args.device, model_name=args.model_name)
        model = BaseSiameseModel(backbone=wrapper, projection_dim=args.projection_dim, use_projector=True).to(args.device)
        print(f"[pairwise-mlp] using {args.backbone} backbone with projection_dim={args.projection_dim}")

    # 2) Load pairs, compute embeddings
    df = load_pairs_dataframe(args.pairs, args.left_col, args.right_col, args.label_col)
    labels = df[args.label_col].astype(int).to_numpy()
    eL, eR, valid = compute_pairwise_embeddings(
        model=model,
        left=df[args.left_col].astype(str).tolist(),
        right=df[args.right_col].astype(str).tolist(),
        batch_size=args.batch_size,
    )
    if valid.sum() == 0:
        print("[pairwise-mlp] No valid pairs; aborting.")
        return
    print(f"[pairwise-mlp] computed embeddings for {valid.sum()}/{len(df)} valid pairs.")

    dfv = df.loc[valid].reset_index(drop=True)
    y = labels[valid]
    X = pairwise_features(eL[valid], eR[valid])

    # (optional) save features for reproducibility
    if args.save_features:
        out = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        out["label"] = y
        out.to_parquet(args.save_features) if args.save_features.endswith((".parquet",".pq")) else out.to_csv(args.save_features, index=False)
        print(f"[pairwise-mlp] saved features -> {args.save_features}")

    # 3) Split train/eval (consistent with your other baselines)
    rng = np.random.RandomState(42)
    idx = np.arange(len(X)); rng.shuffle(idx)
    cut = int(args.mlp_split * len(X))
    tr_idx, ev_idx = idx[:cut], idx[cut:]
    Xtr, Ytr = X[tr_idx], y[tr_idx]
    Xev, Yev = X[ev_idx], y[ev_idx]

    # class imbalance handling
    pos = max(1, int(Ytr.sum()))
    neg = max(1, int((Ytr == 0).sum()))
    pos_weight = neg / pos

    # 4) Train or load MLP
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    if args.load_head:
        print(f"[pairwise-mlp] loading head from {args.load_head}")
        from model_utils.heads.pairwise_mlp import PairwiseMLP
        model_head = PairwiseMLP(in_dim=X.shape[1], hidden=args.mlp_hidden, dropout=args.mlp_dropout)
        model_head.load_state_dict(torch.load(args.load_head, map_location="cpu"))
        model_head = model_head.to(device).eval()
        with torch.no_grad():
            logits = model_head(torch.from_numpy(Xev).to(device)).cpu().numpy()
    else:
        print(f"[pairwise-mlp] training head on {len(Ytr)} pairs, eval on {len(Yev)} pairs")
        model_head, best_auc = train_pairwise_mlp(
            Xtr, Ytr, Xev, Yev,
            hidden=args.mlp_hidden, dropout=args.mlp_dropout,
            lr=args.mlp_lr, epochs=args.mlp_epochs,
            weight_decay=args.mlp_weight_decay,
            pos_weight=pos_weight, device=device,
        )
        print(f"[pairwise-mlp] best val ROC-AUC during training: {best_auc:.4f}")
        if args.save_head:
            torch.save(model_head.state_dict(), args.save_head)
            print(f"[pairwise-mlp] saved head -> {args.save_head}")
        model_head.eval()
        with torch.no_grad():
            logits = model_head(torch.from_numpy(Xev).to(device)).cpu().numpy()

    # 5) Evaluate on eval split
    probs = 1 / (1 + np.exp(-logits))
    from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, roc_curve
    auc = roc_auc_score(Yev, probs)
    thr, tpr, fpr, youden = best_youden_threshold(Yev, probs)
    pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(Yev, pred).ravel()
    acc = accuracy_score(Yev, pred)

    print("\n=== Pairwise-MLP Summary ===")
    print(f"Eval size: {len(Yev)}")
    print(f"ROC_AUC: {auc:.6f}")
    print(f"Best-Youden threshold: {thr:.4f} (TPR={tpr:.4f}, FPR={fpr:.4f}, Youden={youden:.4f})")
    print("Confusion Matrix [TN FP; FN TP]:")
    print(np.array([[tn, fp], [fn, tp]]))
    print(f"Accuracy: {acc:.4f}")
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    if args.roc_path:
        save_roc_plot(Yev, probs, args.roc_path, title="ROC (pairwise-mlp)")

    # optional: save per-pair results aligned to eval split
    if args.save_results:
        out = dfv.iloc[ev_idx].copy()
        out["prob"] = probs
        out["pred"] = pred
        out.rename(columns={
            args.left_col: "left",
            args.right_col: "right",
            args.label_col: "label"
        }, inplace=True)
        ext = os.path.splitext(args.save_results)[-1].lower()
        os.makedirs(os.path.dirname(args.save_results), exist_ok=True) if os.path.dirname(args.save_results) else None
        (out.to_parquet(args.save_results, index=False)
         if ext in (".parquet", ".pq") else
         out.to_csv(args.save_results, index=False))
        print(f"[info] Saved eval results to {args.save_results}")


def run_optuna(args: argparse.Namespace) -> None:
    """
    Hyperparameter optimization training using your existing UnifiedHyperparameterOptimizer.
    Mirrors the interface from your older script so your previous command keeps working.
    """
    # Validate required inputs
    if not args.training_filepath or not args.test_filepath:
        raise SystemExit("--mode optuna requires --training_filepath and --test_filepath")

    os.makedirs(args.log_dir, exist_ok=True)

    optimizer = UnifiedHyperparameterOptimizer(args.backbone, device=args.device, log_dir=args.log_dir)

    opt_params = {
        "n_trials": args.n_trials,
        "sampler": args.sampler,
        "pruner": (None if args.pruner == "none" else args.pruner),
        "study_name": args.study_name,
        "epochs": args.epochs,
    }

    # This matches your old call signature
    _ = optimizer.optimize(
        method="optuna",
        training_filepath=args.training_filepath,
        test_filepath=args.test_filepath,
        mode=args.model_type,
        loss_type=args.loss_type,
        medium_filepath=args.medium_filepath,
        easy_filepath=args.easy_filepath,
        curriculum=args.curriculum,
        validate_filepath=args.validate_filepath,
        **opt_params,
    )

    print(f"[optuna] Completed. Check logs/artifacts in: {args.log_dir}")


# -----------
# Main driver
# -----------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified entry point for baseline, evaluate_saved, and optuna.")

    # Modes
    p.add_argument("--mode", type=str, required=True,
                   choices=["baseline", "evaluate_saved", "optuna"],
                   help="baseline: zero/one-shot eval; evaluate_saved: load a trained checkpoint; optuna: train with HPO.")

    # Data for evaluation/baseline (pairs only needed for those modes)
    p.add_argument("--pairs", type=str, default=None,
                   help="CSV/TSV/Parquet with left/right/label columns (required for baseline/evaluate_saved).")
    p.add_argument("--left-col", type=str, default="fraudulent_name")
    p.add_argument("--right-col", type=str, default="real_name")
    p.add_argument("--label-col", type=str, default="label")

    # Backbone selection
    p.add_argument("--backbone", type=str, default="clip",
                   help="Backbone: clip, siglip, flava, coca, or precomputed.")
    p.add_argument("--model-name", type=str, default=None,
                   help="Optional HF/OpenAI model name for certain backbones.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Precomputed-specific
    p.add_argument("--npz", nargs="+", default=None,
                   help="(precomputed) One or more NPZ files mapping word -> 1D vector.")
    p.add_argument("--pc-norm", action="store_true",
                   help="(precomputed) L2-normalize vectors on load/encode.")
    p.add_argument("--pc-case-sensitive", action="store_true",
                   help="(precomputed) Treat NPZ keys as case-sensitive (default lowercases).")
    p.add_argument("--pc-non-strict", action="store_true",
                   help="(precomputed) Missing words map to zeros instead of error.")

    # Projector / batching
    p.add_argument("--projection-dim", type=int, default=128,
                   help="Projection size if projector is used (ignored for precomputed).")
    p.add_argument("--batch-size", type=int, default=2048,
                   help="Batch size for encoding/scoring.")

    # evaluate_saved
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to a saved model checkpoint (evaluate_saved).")

    # Outputs (common)
    p.add_argument("--save-results", type=str, default=None,
                   help="Optional CSV/Parquet to save per-pair scores.")
    p.add_argument("--roc-path", type=str, default=None,
                   help="If set, save a TPR-vs-FPR ROC plot to this path.")

    # Pairwise-MLP head (optional, baseline mode)
    p.add_argument("--pairwise-mlp", action="store_true",
                   help="Use an MLP head on pairwise features instead of cosine.")
    p.add_argument("--mlp-hidden", type=int, default=64)
    p.add_argument("--mlp-dropout", type=float, default=0.10)
    p.add_argument("--mlp-epochs", type=int, default=30)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    p.add_argument("--mlp-split", type=float, default=0.40)
    p.add_argument("--save-head", type=str, default=None)
    p.add_argument("--load-head", type=str, default=None)
    p.add_argument("--save-features", type=str, default=None)

    # --- NEW: training / validation / optimization args for optuna mode ---
    p.add_argument("--training_filepath", type=str, default=None,
                   help="Path to training data (CSV/Parquet). Required for optuna.")
    p.add_argument("--test_filepath", type=str, default=None,
                   help="Path to test/eval data (CSV/Parquet). Required for optuna.")
    p.add_argument("--validate_filepath", type=str, default=None,
                   help="Optional validation set used during/after training.")
    p.add_argument("--model_type", type=str,
                   choices=["pair", "triplet", "supcon", "infonce"], default="infonce")
    p.add_argument("--loss_type", type=str,
                   choices=["cosine", "euclidean", "hybrid", "supcon", "infonce"], default="infonce")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--curriculum", type=str, default=None)
    p.add_argument("--medium_filepath", type=str, default=None)
    p.add_argument("--easy_filepath", type=str, default=None)

    # HPO controls (Optuna)
    p.add_argument("--n_trials", type=int, default=10)
    p.add_argument("--sampler", type=str, choices=["tpe", "random", "cmaes"], default="tpe")
    p.add_argument("--pruner", type=str, choices=["median", "hyperband", "none"], default="median")
    p.add_argument("--study_name", type=str, default=None)

    # Logging
    p.add_argument("--log-dir", type=str, default="./logs",
                   help="Directory to write checkpoints/results.")

    return p


def save_roc_plot(y_true: np.ndarray, y_score: np.ndarray, out_path: str, title: str = "TPR vs FPR (ROC)") -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()

    # ensure parent dir exists if provided
    if out_path and os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[info] Saved ROC curve -> {out_path}")



def run_baseline(args: argparse.Namespace) -> None:
    # Build wrapper
    if args.backbone.lower() == "precomputed":
        if not args.npz:
            raise SystemExit("When --backbone precomputed, you must pass --npz <files...>")
        wrapper = ModelFactory.create_model(
            "precomputed",
            npz_paths=args.npz,
            lowercase_keys=not args.pc_case_sensitive,
            normalize=args.pc_norm,
            strict=not args.pc_non_strict,
        )
        # Skip projector entirely for precomputed vectors
        model = BaseSiameseModel(backbone=wrapper, projection_dim=wrapper.embedding_dim, use_projector=False)
    else:
        # Standard backbone path (requires corresponding wrappers present)
        wrapper = ModelFactory.create_model(args.backbone, device=args.device, model_name=args.model_name)
        model = BaseSiameseModel(backbone=wrapper, projection_dim=args.projection_dim, use_projector=True)
        model = model.to(args.device)  # in case wrapper returns device tensors

    # Load pairs
    df = load_pairs_dataframe(args.pairs, args.left_col, args.right_col, args.label_col)
    labels = df[args.label_col].astype(int).to_numpy()

    # Compute pairwise similarities
    scores, valid = compute_pairwise_scores(
        model=model,
        left=df[args.left_col].astype(str).tolist(),
        right=df[args.right_col].astype(str).tolist(),
        batch_size=args.batch_size,
    )

    total, good = len(df), int(valid.sum())
    print(f"[baseline] pairs={total}  valid={good}")

    if good == 0:
        print("No valid pairs had embeddings on both sides; nothing to evaluate.")
        _maybe_save_results(args, df, scores)
        return

    v_labels = labels[valid]
    v_scores = scores[valid]

    if np.unique(v_labels).size < 2:
        print("Valid pairs do not contain both label classes; cannot compute ROC AUC or thresholds.")
        _maybe_save_results(args, df, scores)
        return

    auc = roc_auc_score(v_labels, v_scores)
    thrs = pick_thresholds(v_labels, v_scores)
    m_y = metrics_at_threshold(v_labels, v_scores, thrs["youden"])
    m_a = metrics_at_threshold(v_labels, v_scores, thrs["max_acc"])

    print("\n=== Summary Metrics (baseline) ===")
    print(f"ROC_AUC: {auc:.6f}")
    print("\n-- Best-Youden --")
    for k, v in m_y.items():
        print(f"{k}: {v}")
    print("\n-- Best-Accuracy --")
    for k, v in m_a.items():
        print(f"{k}: {v}")


    if args.roc_path:
        save_roc_plot(v_labels, v_scores, args.roc_path, title=f"ROC (baseline, {args.backbone})")

    _maybe_save_results(args, df, scores)


def run_evaluate_saved(args: argparse.Namespace) -> None:
    """
    Generic checkpoint loader. This assumes your checkpoint contains:
      - 'state_dict': model weights for BaseSiameseModel
      - 'backbone_name', 'projection_dim', and (optionally) 'model_name'
    Adjust as needed for your actual checkpoint format.
    """
    if not args.checkpoint:
        raise SystemExit("--mode evaluate_saved requires --checkpoint")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    backbone_name = ckpt.get("backbone_name", None)
    projection_dim = ckpt.get("projection_dim", 128)
    model_name = ckpt.get("model_name", None)

    if backbone_name is None:
        raise RuntimeError("Checkpoint missing 'backbone_name'. Please include it when saving checkpoints.")

    if backbone_name.lower() == "precomputed":
        # Precomputed + checkpoint doesn't make sense; skip projector and don't load state_dict.
        if not args.npz:
            raise SystemExit("evaluate_saved with backbone=precomputed requires --npz to supply vectors.")
        wrapper = ModelFactory.create_model(
            "precomputed",
            npz_paths=args.npz,
            lowercase_keys=not args.pc_case_sensitive,
            normalize=args.pc_norm,
            strict=not args.pc_non_strict,
        )
        model = BaseSiameseModel(backbone=wrapper, projection_dim=wrapper.embedding_dim, use_projector=False)
    else:
        wrapper = ModelFactory.create_model(backbone_name, device=args.device, model_name=model_name)
        model = BaseSiameseModel(backbone=wrapper, projection_dim=int(projection_dim), use_projector=True)
        sd = ckpt.get("state_dict", None)
        if sd is None:
            raise RuntimeError("Checkpoint missing 'state_dict'.")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[warn] load_state_dict: missing={missing}, unexpected={unexpected}")
        model = model.to(args.device)

    # Evaluate exactly like baseline
    df = load_pairs_dataframe(args.pairs, args.left_col, args.right_col, args.label_col)
    labels = df[args.label_col].astype(int).to_numpy()

    scores, valid = compute_pairwise_scores(
        model=model,
        left=df[args.left_col].astype(str).tolist(),
        right=df[args.right_col].astype(str).tolist(),
        batch_size=args.batch_size,
    )

    total, good = len(df), int(valid.sum())
    print(f"[evaluate_saved] pairs={total}  valid={good}")

    if good == 0:
        print("No valid pairs had embeddings on both sides; nothing to evaluate.")
        _maybe_save_results(args, df, scores)
        return

    v_labels = labels[valid]
    v_scores = scores[valid]

    if np.unique(v_labels).size < 2:
        print("Valid pairs do not contain both label classes; cannot compute ROC AUC or thresholds.")
        _maybe_save_results(args, df, scores)
        return

    auc = roc_auc_score(v_labels, v_scores)
    thrs = pick_thresholds(v_labels, v_scores)
    m_y = metrics_at_threshold(v_labels, v_scores, thrs["youden"])
    m_a = metrics_at_threshold(v_labels, v_scores, thrs["max_acc"])

    print("\n=== Summary Metrics (evaluate_saved) ===")
    print(f"ROC_AUC: {auc:.6f}")
    print("\n-- Best-Youden --")
    for k, v in m_y.items():
        print(f"{k}: {v}")
    print("\n-- Best-Accuracy --")
    for k, v in m_a.items():
        print(f"{k}: {v}")

    if args.roc_path:
        save_roc_plot(v_labels, v_scores, args.roc_path, title=f"ROC (evaluate_saved, {backbone_name})")

    _maybe_save_results(args, df, scores)


def _maybe_save_results(args: argparse.Namespace, df: pd.DataFrame, scores: np.ndarray) -> None:
    if not args.save_results:
        return
    out = df.copy()
    out["similarity"] = scores
    out.rename(columns={
        args.left_col: "left",
        args.right_col: "right",
        args.label_col: "label"
    }, inplace=True)
    ext = os.path.splitext(args.save_results)[-1].lower()
    os.makedirs(os.path.dirname(args.save_results), exist_ok=True) if os.path.dirname(args.save_results) else None
    if ext in (".parquet", ".pq"):
        out.to_parquet(args.save_results, index=False)
    else:
        out.to_csv(args.save_results, index=False)
    print(f"[info] Saved per-pair results to {args.save_results}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "baseline":
        if args.pairwise_mlp:
            run_pairwise_mlp(args)
        else:
            run_baseline(args)
    elif args.mode == "evaluate_saved":
        run_evaluate_saved(args)
    elif args.mode == "optuna":
        run_optuna(args)
    else:
        raise SystemExit(f"Unsupported mode: {args.mode}")

if __name__ == "__main__":
    main()