import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import token_set_ratio

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