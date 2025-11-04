# Use precomputed SigLip embeddings and cosine similarity to compute TPR at FPR <= 1%
# This block will run immediately if the script is executed and then exit so the rest
# of the original file (token_set_ratio based) is not used.
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import token_set_ratio
from tqdm import tqdm

if __name__ == "__main__":

    PARQUET_PATH = Path("data/processed/validate_pairs_ref_10k.parquet")
    EMB_PATH = Path("data/embeddings/siglip_glyphs/image_embeddings_validate.npz")
    MAX_FPR = 0.01

    def load_embeddings(npz_path: Path):
        if not npz_path.exists():
            raise SystemExit(f"Embeddings file not found: {npz_path}")
        data = np.load(npz_path, allow_pickle=True)
        ids = []
        emb = []
        for k in tqdm(data.files):
                key = k.split(".")[0]
                v = data[k].astype(np.float32)
                ids.append(key)
                emb.append(v)

        # Ensure proper numpy arrays
        ids = np.asarray(ids).astype(str)
        emb = np.asarray(emb).astype(float)

        # Normalize embeddings to unit vectors (safe)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb_unit = emb / norms

        id2vec = {ids[i]: emb_unit[i] for i in range(len(ids))}
        return id2vec

    def load_pairs(parquet_path: Path):
        if not parquet_path.exists():
            raise SystemExit(f"Pairs file not found: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        df.columns = [str(c).strip() for c in df.columns]
        # build pairs
        pairs = pd.DataFrame({
            "name1": df["fraudulent_name"].fillna("").astype(str),
            "name2": df["real_name"].fillna("").astype(str),
            "label": df["label"].astype(str).str.lower()
        })
        # normalize label to 0/1
        positive_vals = {"1", "true", "yes"}
        negative_vals = {"0", "false", "no"}
        def map_label(v):
            v = str(v).strip().lower()
            if v in positive_vals:
                return 1
            if v in negative_vals:
                return 0
            try:
                return int(float(v))
            except Exception:
                raise ValueError(f"Unrecognized label value: {v}")
        pairs["label"] = pairs["label"].apply(map_label).astype(int)
        return pairs

    def compute_scores_from_embeddings(pairs: pd.DataFrame, id2vec: dict):
        scores = []
        labels = []
        skipped = 0
        for _, r in pairs.iterrows():
            n1 = str(r["name1"])
            n2 = str(r["name2"])
            v1 = id2vec.get(n1)
            v2 = id2vec.get(n2)
            if v1 is None or v2 is None:
                skipped += 1
                continue
            # cosine similarity (unit vectors => dot product)
            # flattening to 1D
            v1 = np.asarray(v1).reshape(-1)
            v2 = np.asarray(v2).reshape(-1)
            v1 = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 0 else v1
            v2 = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 0 else v2
            sim = float(np.dot(v1, v2)) 
            scores.append(sim)
            labels.append(int(r["label"]))
        if len(scores) == 0:
            raise SystemExit("No pairs with both precomputed embeddings were found. Nothing to evaluate.")
        return np.array(scores, dtype=float), np.array(labels, dtype=int), skipped

    def select_threshold(scores: np.ndarray, labels: np.ndarray, max_fpr: float = 0.01):
        thresholds = np.linspace(-1.0, 1.0, 2001)  # step 0.001
        best = {"threshold": None, "tpr": 0.0, "fpr": 1.0}
        P = int((labels == 1).sum())
        N = int((labels == 0).sum())
        if P == 0 or N == 0:
            raise ValueError("Need both positive and negative examples to compute rates.")

        tpr_list = []
        fpr_list = []
        print(scores)
        for t in thresholds:
            preds = (scores >= t).astype(int)
            tp = int(((preds == 1) & (labels == 1)).sum())
            fp = int(((preds == 1) & (labels == 0)).sum())
            tpr = tp / P
            fpr = fp / N
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            if fpr <= max_fpr and tpr > best["tpr"]:
                best = {"threshold": t, "tpr": tpr, "fpr": fpr}

        # Save numeric data for later inspection
        try:
            np.savez("tpr_fpr_cosine.npz", thresholds=thresholds, tpr=np.array(tpr_list), fpr=np.array(fpr_list))
        except Exception:
            pass
        try:
            pd.DataFrame({
            "threshold": thresholds,
            "tpr": np.array(tpr_list),
            "fpr": np.array(fpr_list)
            }).to_csv("tpr_fpr_cosine.csv", index=False)
        except Exception:
            pass

        # Plot TPR vs FPR curve and mark selected threshold (if any)
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 5))
            plt.plot(fpr_list, tpr_list, color="C0", lw=1.5, label="ROC curve (cosine thresholds)")
            # mark selected point
            if best["threshold"] is not None:
            # find nearest index to selected threshold
                idx = int(np.argmin(np.abs(thresholds - best["threshold"])))
            plt.scatter([fpr_list[idx]], [tpr_list[idx]], color="red", zorder=5, label=f"Selected (thr={best['threshold']:.3f})")
            plt.annotate(f"thr={best['threshold']:.3f}\nTPR={best['tpr']:.4f}\nFPR={best['fpr']:.6f}",
                     (fpr_list[idx], tpr_list[idx]),
                     textcoords="offset points", xytext=(6, -18), fontsize=8)
            plt.xlabel("False Positive Rate (FPR)")
            plt.ylabel("True Positive Rate (TPR)")
            plt.title(f"TPR vs FPR (cosine similarity) — target FPR <= {max_fpr:.3f}")
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)
            plt.grid(alpha=0.3)
            plt.legend(loc="lower right", fontsize=8)
            plt.tight_layout()
            plt.savefig("tpr_vs_fpr_cosine.png", dpi=150)
            plt.close()
        except Exception:
            # plotting is optional; ignore failures (e.g., headless environment without matplotlib)
            pass

        return best

    # Run pipeline
    id2vec = load_embeddings(EMB_PATH)
    pairs = load_pairs(PARQUET_PATH)
    scores, labels, skipped = compute_scores_from_embeddings(pairs, id2vec)

    try:
        result = select_threshold(scores, labels, max_fpr=MAX_FPR)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    if result["threshold"] is None:
        print("No threshold found that achieves FPR <= 1% on available pairs.")
        # find threshold with the smallest positive FPR (>0); among ties pick highest TPR
        thresholds = np.linspace(-1.0, 1.0, 2001)
        P = int((labels == 1).sum()); N = int((labels == 0).sum())
        best_minpos = {"threshold": None, "tpr": 0.0, "fpr": None}
        min_pos_fpr = None
        for t in thresholds:
            preds = (scores >= t).astype(int)
            tp = int(((preds == 1) & (labels == 1)).sum())
            fp = int(((preds == 1) & (labels == 0)).sum())
            tpr = tp / P
            fpr = fp / N
            if fpr > 0.0:
                if (min_pos_fpr is None) or (fpr < min_pos_fpr):
                    min_pos_fpr = fpr
                    best_minpos.update({"threshold": t, "tpr": tpr, "fpr": fpr})
                elif fpr == min_pos_fpr and tpr > best_minpos["tpr"]:
                    best_minpos.update({"threshold": t, "tpr": tpr, "fpr": fpr})

        if best_minpos["threshold"] is not None:
            print(f"Selected threshold (cosine) at lowest positive FPR = {best_minpos['threshold']:.3f}")
            print(f"Fraud detection rate (TPR) at that lowest positive FPR: {best_minpos['tpr']:.4f}")
            print(f"Observed FPR at that threshold: {best_minpos['fpr']:.6f}")
        else:
            # fallback: no positive FPR found (all FPR == 0). report best TPR (any FPR)
            thresholds = np.linspace(-1.0, 1.0, 2001)
            best_any = {"threshold": None, "tpr": 0.0, "fpr": 1.0}
            for t in thresholds:
                preds = (scores >= t).astype(int)
                tp = int(((preds == 1) & (labels == 1)).sum())
                fp = int(((preds == 1) & (labels == 0)).sum())
                tpr = tp / P
                fpr = fp / N
                if tpr > best_any["tpr"]:
                    best_any.update({"threshold": t, "tpr": tpr, "fpr": fpr})
            print(f"No positive FPR observed; reporting best threshold by TPR: {best_any['threshold']:.3f}, TPR={best_any['tpr']:.4f}, FPR={best_any['fpr']:.6f}")
    else:
        print(f"Selected threshold (cosine) = {result['threshold']:.3f}")
        print(f"Fraud detection rate (TPR) at FPR <= 1%: {result['tpr']:.4f}")
        print(f"Observed FPR at that threshold: {result['fpr']:.6f}")

    print(f"Pairs evaluated (with embeddings): {len(scores)}, pairs skipped (missing embeddings): {skipped}")
    sys.exit(0)
    

# /c:/Users/hbori/Documents/Itau_env/Itau-group2/scripts/tsr_results.py
"""
Compute fraud detection rate (TPR) using token_set_ratio on pairs file.
Select the highest similarity threshold such that false positive rate (FPR) <= 1%.

Usage:
    python tsr_results.py [path_to_pairs_file]

Default path: data/processed/validate_pairs_ref_10k
"""


try:
    import pandas as pd
except Exception as e:
    raise SystemExit("pandas is required. Install with: pip install pandas") from e

try:
    from rapidfuzz.fuzz import token_set_ratio
except Exception as e:
    raise SystemExit("rapidfuzz is required. Install with: pip install rapidfuzz") from e

import sys
from pathlib import Path
import numpy as np
import pandas as pd

def _load_pairs(path: Path) -> pd.DataFrame:
    # read parquet 
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        raise SystemExit(f"Error reading parquet file: {e}") from e

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # If file already has label column detect it; else assume three columns: name1, name2, label
    # Find candidate name columns (string-like) and a label column (binary)
    label_col = None
    name_cols = []

    # detect label column by checking values
    for c in df.columns:
        vals = df[c].dropna().astype(str).str.lower().unique()[:10]
        # common binary values
        if all(v in {"0", "1", "true", "false", "yes", "no"} for v in vals):
            label_col = c
            break

    # identify name columns as columns with mostly non-numeric strings
    for c in df.columns:
        if c == label_col:
            continue
        sample = df[c].dropna().astype(str).head(50)
        # consider as name-like if any alpha character present in samples
        if any(s.strip() != "" and any(ch.isalpha() for ch in s) for s in sample):
            name_cols.append(c)

    # If we didn't find label column but we have exactly 3 columns, assume third is label
    if label_col is None and len(df.columns) == 3:
        label_col = df.columns[2]
        name_cols = [df.columns[0], df.columns[1]]

    # If we still don't have two name columns, try first two columns
    if len(name_cols) < 2:
        name_cols = list(df.columns[:2])

    if len(name_cols) < 2:
        raise SystemExit("Couldn't infer two name columns from file. Please provide a file with at least two name columns and a binary label column.")

    if label_col is None:
        raise SystemExit("Couldn't infer label column. Label must be binary (0/1, true/false, yes/no).")

    # Build clean DataFrame with name1, name2, label
    pairs = pd.DataFrame({
        "name1": df[name_cols[0]].fillna("").astype(str),
        "name2": df[name_cols[1]].fillna("").astype(str),
        "label": df[label_col].astype(str).str.lower()
    })

    # normalize label to 0/1 ints (1 => spoof/fraud/positive)
    positive_vals = {"1", "true", "yes"}
    negative_vals = {"0", "false", "no"}
    def map_label(v):
        v = v.strip().lower()
        if v in positive_vals:
            return 1
        if v in negative_vals:
            return 0
        # try numeric parse
        try:
            return int(float(v))
        except Exception:
            raise ValueError(f"Unrecognized label value: {v}")
    pairs["label"] = pairs["label"].apply(map_label).astype(int)
    return pairs


def compute_scores(pairs: pd.DataFrame) -> np.ndarray:
    # compute token_set_ratio for each pair
    scores = pairs.apply(lambda r: token_set_ratio(r["name1"], r["name2"]), axis=1).to_numpy(dtype=float)
    return scores


def select_threshold(scores: np.ndarray, labels: np.ndarray, max_fpr: float = 0.01):
    # search thresholds between 0 and 100 (inclusive) with step 0.1 for precision
    thresholds = np.linspace(0, 100, 1001)
    best = {"threshold": None, "tpr": 0.0, "fpr": 1.0}
    P = int((labels == 1).sum())
    N = int((labels == 0).sum())
    if P == 0 or N == 0:
        raise ValueError("Need both positive and negative examples to compute rates.")

    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        tpr = tp / P
        fpr = fp / N
        if fpr <= max_fpr and tpr > best["tpr"]:
            best.update({"threshold": t, "tpr": tpr, "fpr": fpr})
    return best


def main(path: str):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    pairs = _load_pairs(path)
    scores = compute_scores(pairs)
    labels = pairs["label"].to_numpy(dtype=int)
    result = select_threshold(scores, labels, max_fpr=0.01)
    if result["threshold"] is None:
        print("No threshold found that achieves FPR <= 1%.")
        print(f"Best achievable (ignoring FPR constraint): report top TPR/FPR pairs instead.")
        # Optionally report maximum TPR at any FPR
        # compute best by TPR
        thresholds = np.linspace(0, 100, 1001)
        best_any = {"threshold": None, "tpr": 0.0, "fpr": 1.0}
        P = int((labels == 1).sum()); N = int((labels == 0).sum())
        for t in thresholds:
            preds = (scores >= t).astype(int)
            tp = int(((preds == 1) & (labels == 1)).sum())
            fp = int(((preds == 1) & (labels == 0)).sum())
            tpr = tp / P
            fpr = fp / N
            if tpr > best_any["tpr"]:
                best_any.update({"threshold": t, "tpr": tpr, "fpr": fpr})
        print(f"Best threshold: {best_any['threshold']:.1f}, TPR={best_any['tpr']:.4f}, FPR={best_any['fpr']:.6f}")
    else:
        print(f"Selected threshold = {result['threshold']:.1f}")
        print(f"Fraud detection rate (TPR) at FPR <= 1%: {result['tpr']:.4f}")
        print(f"Observed FPR at that threshold: {result['fpr']:.6f}")


if __name__ == "__main__":
    fp = sys.argv[1] if len(sys.argv) > 1 else "data/processed/validate_pairs_ref_10k.parquet"
    main(fp)