# scripts/evaluation/precomputed_eval.py
import argparse
import os
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, confusion_matrix


class PrecomputedEmbeddingModel:
    """
    Minimal shim that exposes `.encode(texts)` and returns vectors from NPZ maps.

    NPZ format: each file is a dict-like mapping `word -> 1D embedding`.
    Assumes vectors are already L2-normalized (set --normalize if not).
    """

    def __init__(
        self,
        npz_paths: Sequence[str],
        normalize: bool = False,
        lowercase_keys: bool = True,
        strict: bool = True,
    ):
        self.normalize = normalize
        self.lowercase_keys = lowercase_keys
        self.strict = strict
        self._store: Dict[str, np.ndarray] = {}

        for p in npz_paths:
            data = np.load(p, allow_pickle=True)
            for key in data.files:
                word = key.lower() if lowercase_keys else key
                vec = data[key]
                vec = np.asarray(vec).squeeze()
                if vec.ndim != 1:
                    raise ValueError(f"Embedding for key '{key}' in {p} must be 1D, got {vec.shape}")
                if normalize:
                    n = np.linalg.norm(vec)
                    if n > 0:
                        vec = vec / n
                self._store[word] = vec.astype(np.float32)

        if not self._store:
            raise ValueError("No embeddings loaded from provided NPZ paths.")

        self.embedding_dim = int(next(iter(self._store.values())).shape[-1])

    def encode(self, texts: Iterable[str]) -> torch.Tensor:
        out: List[torch.Tensor] = []
        for t in texts:
            key = t.lower() if self.lowercase_keys else t
            vec = self._store.get(key)
            if vec is None:
                if self.strict:
                    raise KeyError(f"Missing embedding for word: {t!r}")
                vec = np.zeros(self.embedding_dim, dtype=np.float32)
            out.append(torch.from_numpy(vec).unsqueeze(0))
        result = torch.cat(out, dim=0)  # [N, D]
        if self.normalize:
            result = F.normalize(result, dim=1)
        return result


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
        raise ValueError(f"Pairs file missing columns: {missing}. Found: {list(df.columns)}")
    return df[[left_col, right_col, label_col]].copy()


def compute_scores(
    model: PrecomputedEmbeddingModel,
    left: List[str],
    right: List[str],
    batch_size: int,
) -> np.ndarray:
    """Return cosine similarities for each (left_i, right_i) pair."""
    # Since vectors are normalized, cosine == dot product
    sims = np.empty(len(left), dtype=np.float32)
    N = len(left)
    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        L = model.encode(left[i:j]).numpy()   # [B, D]
        R = model.encode(right[i:j]).numpy()  # [B, D]
        sims[i:j] = np.einsum("bd,bd->b", L, R)  # batched dot product
    return sims


def pick_thresholds(labels: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Return Youden-J and max-accuracy thresholds based on ROC."""
    fpr, tpr, thresh = roc_curve(labels, scores)
    youden_idx = np.argmax(tpr - fpr)
    thr_youden = float(thresh[youden_idx])

    # max-accuracy search on same thresholds grid
    preds_grid = (scores[:, None] >= thresh[None, :])
    accs = (preds_grid == labels[:, None]).mean(axis=0)
    thr_maxacc = float(thresh[int(np.argmax(accs))])

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


def main():
    parser = argparse.ArgumentParser(description="Evaluate precomputed SigLIP-glyph embeddings without training.")
    parser.add_argument("--pairs", type=str, required=True,
                        help="CSV/TSV/Parquet with columns for left/right/label.")
    parser.add_argument("--left-col", type=str, default="fraudulent_name",
                        help="Column name for left/query word.")
    parser.add_argument("--right-col", type=str, default="real_name",
                        help="Column name for right/reference word.")
    parser.add_argument("--label-col", type=str, default="label",
                        help="Column name for binary label (0/1).")
    parser.add_argument("--npz", type=str, nargs="+", required=True,
                        help="One or more .npz files mapping word -> 1D embedding.")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Batch size for encoding/scoring.")
    parser.add_argument("--normalize", action="store_true",
                        help="Apply L2-normalization on load/encode (use only if NPZ not normalized).")
    parser.add_argument("--case-sensitive", action="store_true",
                        help="Treat NPZ keys as case-sensitive (default lowers them).")
    parser.add_argument("--non-strict", action="store_true",
                        help="Use zero vector for missing words instead of raising.")
    parser.add_argument("--save-results", type=str, default=None,
                        help="Optional CSV/Parquet path to save per-pair results.")
    args = parser.parse_args()

    df = load_pairs_dataframe(args.pairs, args.left_col, args.right_col, args.label_col)
    labels = df[args.label_col].astype(int).to_numpy()

    model = PrecomputedEmbeddingModel(
        npz_paths=args.npz,
        normalize=args.normalize,
        lowercase_keys=not args.case_sensitive,
        strict=not args.non_strict,
    )

    scores = compute_scores(
        model=model,
        left=df[args.left_col].astype(str).tolist(),
        right=df[args.right_col].astype(str).tolist(),
        batch_size=args.batch_size,
    )

    # Metrics
    auc = roc_auc_score(labels, scores)
    thrs = pick_thresholds(labels, scores)
    m_youden = metrics_at_threshold(labels, scores, thrs["youden"])
    m_acc = metrics_at_threshold(labels, scores, thrs["max_acc"])

    print("\n=== Summary Metrics ===")
    print(f"ROC_AUC: {auc:.6f}")
    print("\n-- Best-Youden --")
    for k, v in m_youden.items():
        print(f"{k}: {v}")
    print("\n-- Best-Accuracy --")
    for k, v in m_acc.items():
        print(f"{k}: {v}")

    # Optional per-pair output
    if args.save_results:
        out = df.copy()
        out["similarity"] = scores
        # normalized column names for convenience
        out.rename(columns={
            args.left_col: "left",
            args.right_col: "right",
            args.label_col: "label"
        }, inplace=True)
        ext = os.path.splitext(args.save_results)[-1].lower()
        if ext in (".parquet", ".pq"):
            out.to_parquet(args.save_results, index=False)
        else:
            out.to_csv(args.save_results, index=False)


if __name__ == "__main__":
    main()
