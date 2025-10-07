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


# -----------
# Main driver
# -----------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified entry point for baseline and evaluate_saved.")

    # Modes
    p.add_argument("--mode", type=str, required=True, choices=["baseline", "evaluate_saved"],
                   help="baseline: zero/one-shot eval without training; evaluate_saved: load a trained checkpoint and evaluate.")

    # Pairs / columns
    p.add_argument("--pairs", type=str, required=True, help="CSV/TSV/Parquet with left/right/label columns.")
    p.add_argument("--left-col", type=str, default="fraudulent_name", help="Column name for left/query word.")
    p.add_argument("--right-col", type=str, default="real_name", help="Column name for right/reference word.")
    p.add_argument("--label-col", type=str, default="label", help="Column name for binary label (0/1).")

    # Backbone selection
    p.add_argument("--backbone", type=str, default="clip",
                   help="Backbone name: clip, siglip, flava, coca, or precomputed (glyph-embedding NPZ).")
    p.add_argument("--model-name", type=str, default=None, help="Optional HF/openai model name for certain backbones.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu.")

    # Precomputed-specific
    p.add_argument("--npz", nargs="+", default=None,
                   help="(precomputed) One or more NPZ files mapping word -> 1D vector.")
    p.add_argument("--pc-norm", action="store_true",
                   help="(precomputed) L2-normalize vectors on load/encode (use only if NPZ not already normalized).")
    p.add_argument("--pc-case-sensitive", action="store_true",
                   help="(precomputed) Treat NPZ keys as case-sensitive (default lowercases).")
    p.add_argument("--pc-non-strict", action="store_true",
                   help="(precomputed) Missing words map to zeros instead of error.")

    # Projector / batching
    p.add_argument("--projection-dim", type=int, default=128,
                   help="Projection size if a projector is used (ignored for precomputed).")
    p.add_argument("--batch-size", type=int, default=2048, help="Batch size for encoding/scoring.")

    # Evaluate_saved (checkpointed models)
    p.add_argument("--checkpoint", type=str, default=None, help="Path to a saved model checkpoint (evaluate_saved).")

    # Outputs
    p.add_argument("--save-results", type=str, default=None, help="Optional CSV/Parquet to save per-pair scores.")
    return p


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
        run_baseline(args)
    elif args.mode == "evaluate_saved":
        run_evaluate_saved(args)
    else:
        raise SystemExit(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()