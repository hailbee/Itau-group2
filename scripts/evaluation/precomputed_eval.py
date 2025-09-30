# scripts/evaluation/precomputed_eval.py
import argparse
import os
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

# Reuse the repo's evaluator (cosine similarity + thresholds + metrics)
try:
    from scripts.evaluation.evaluator import Evaluator
except Exception as e:
    raise ImportError(
        "Could not import scripts.evaluation.evaluator.Evaluator. "
        "Make sure you're running from the repo root and that the module exists.\n"
        f"Original error: {e}"
    )


class PrecomputedEmbeddingModel:
    """
    Minimal shim so Evaluator -> EmbeddingExtractor -> model.encode(texts) works.

    Expects one or more .npz files where each file is a mapping:
        word (str) -> embedding (1D np.ndarray)

    Assumptions:
    - Your vectors are ALREADY L2-normalized. If not, set normalize=True.
    - Word lookup is case-insensitive by default.
    """

    def __init__(
        self,
        npz_paths: Sequence[str],
        normalize: bool = False,
        lowercase_keys: bool = True,
        strict: bool = True,
    ):
        """
        :param npz_paths: list/tuple of .npz files to merge (later files win on key clashes)
        :param normalize: if True, L2-normalize vectors on load/encode (your files are already normalized -> False)
        :param lowercase_keys: if True, store and query by lowercase
        :param strict: if True, raise KeyError on missing word; otherwise create a zero vector
        """
        self.normalize = normalize
        self.lowercase_keys = lowercase_keys
        self.strict = strict
        self._store: Dict[str, np.ndarray] = {}

        for p in npz_paths:
            data = np.load(p, allow_pickle=True)
            # interpret each key in the NPZ as a word
            for key in data.files:
                word = key
                if self.lowercase_keys:
                    word = word.lower()
                vec = data[key]
                if vec.ndim != 1:
                    raise ValueError(f"Embedding for key '{key}' in {p} must be 1D, got shape {vec.shape}")
                if self.normalize:
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                self._store[word] = vec.astype(np.float32)

        if not self._store:
            raise ValueError("No embeddings loaded from provided NPZ paths.")

        # cache dim
        any_vec = next(iter(self._store.values()))
        self.embedding_dim = int(any_vec.shape[-1])

    def encode(self, texts: Iterable[str]) -> torch.Tensor:
        """Return a [N, D] float32 tensor of embeddings for the given texts."""
        out: List[torch.Tensor] = []
        for t in texts:
            key = t.lower() if self.lowercase_keys else t
            vec = self._store.get(key)
            if vec is None:
                if self.strict:
                    raise KeyError(f"Missing embedding for word: {t!r}")
                # fallback: zero vector
                vec = np.zeros(self.embedding_dim, dtype=np.float32)
            ten = torch.from_numpy(vec).unsqueeze(0)  # [1, D]
            out.append(ten)
        result = torch.cat(out, dim=0)  # [N, D]
        if self.normalize:
            result = F.normalize(result, dim=1)
        return result


def load_pairs_dataframe(path: str, cols: Dict[str, str]) -> pd.DataFrame:
    """
    Load a pairs dataframe from CSV or Parquet with columns for left, right, label.
    `cols` must map: {"left": <colname>, "right": <colname>, "label": <colname>}
    """
    ext = os.path.splitext(path)[-1].lower()
    if ext in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    elif ext in (".csv", ".tsv"):
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported pairs file type: {ext}. Use .csv, .tsv, or .parquet")

    missing = [c for c in (cols["left"], cols["right"], cols["label"]) if c not in df.columns]
    if missing:
        raise ValueError(f"Pairs file missing required columns: {missing}. Found: {list(df.columns)}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate precomputed SigLIP-glyph word embeddings with existing Evaluator.")
    parser.add_argument("--pairs", type=str, required=True,
                        help="Path to pairs file (CSV/TSV/Parquet) containing columns for left/right/label.")
    parser.add_argument("--left-col", type=str, default="fraudulent_name",
                        help="Column name for left/query word (default: fraudulent_name).")
    parser.add_argument("--right-col", type=str, default="real_name",
                        help="Column name for right/reference word (default: real_name).")
    parser.add_argument("--label-col", type=str, default="label",
                        help="Column name for binary label (default: label).")
    parser.add_argument("--npz", type=str, nargs="+", required=True,
                        help="One or more .npz files mapping word -> 1D embedding. Later files override earlier ones.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for evaluation.")
    parser.add_argument("--normalize", action="store_true",
                        help="If set, L2-normalize vectors on load/encode (use if your npz is NOT already normalized).")
    parser.add_argument("--case-sensitive", action="store_true",
                        help="If set, do NOT lowercase keys/queries.")
    parser.add_argument("--non-strict", action="store_true",
                        help="If set, missing words get a zero vector instead of erroring.")
    parser.add_argument("--save-results", type=str, default=None,
                        help="Optional path to save per-pair results (CSV or Parquet).")
    parser.add_argument("--no-plots", action="store_true", help="Disable ROC/confusion plots.")
    args = parser.parse_args()

    cols = {"left": args.left_col, "right": args.right_col, "label": args.label_col}
    df = load_pairs_dataframe(args.pairs, cols)

    model = PrecomputedEmbeddingModel(
        npz_paths=args.npz,
        normalize=args.normalize,
        lowercase_keys=not args.case_sensitive,
        strict=not args.non_strict,
    )

    evaluator = Evaluator(model=model, batch_size=args.batch_size, model_type="pairs")
    # The evaluator expects columns named as in its default pipeline; we pass the DF as-is,
    # so ensure your Evaluator reads the column names from the DataFrame (most repo versions do).
    results_df, metrics = evaluator.evaluate_dataframe(
        df.rename(columns={
            args.left_col: "fraudulent_name",
            args.right_col: "real_name",
            args.label_col: "label",
        }),
        plot=not args.no_plots
    )

    print("\n=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if args.save_results:
        out_ext = os.path.splitext(args.save_results)[-1].lower()
        if out_ext in (".parquet", ".pq"):
            results_df.to_parquet(args.save_results, index=False)
        else:
            results_df.to_csv(args.save_results, index=False)


if __name__ == "__main__":
    main()
