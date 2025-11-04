import numpy as np

def _safe_norm(x, eps=1e-8):
    return np.sqrt((x * x).sum(axis=-1, keepdims=True) + eps)

def pairwise_features(eL: np.ndarray, eR: np.ndarray) -> np.ndarray:
    """
    Build a compact feature vector from two embedding matrices.
    eL, eR: shape (N, D) float32/float64 numpy arrays.
    Returns: (N, F) float32, where F = 13 (see below).
    """
    assert eL.shape == eR.shape, "eL and eR must have same shape"
    dot = (eL * eR).sum(axis=1)
    nL = _safe_norm(eL)[:, 0]
    nR = _safe_norm(eR)[:, 0]
    cos = dot / (nL * nR + 1e-8)
    angle = np.arccos(np.clip(cos, -1.0, 1.0))
    l2 = np.sqrt(((eL - eR) ** 2).sum(axis=1))
    l1 = np.abs(eL - eR).sum(axis=1)

    eLn = eL / (nL[:, None] + 1e-8)
    eRn = eR / (nR[:, None] + 1e-8)
    nl2 = np.sqrt(((eLn - eRn) ** 2).sum(axis=1))

    eLc = eL - eL.mean(axis=1, keepdims=True)
    eRc = eR - eR.mean(axis=1, keepdims=True)
    corr = (eLc * eRc).sum(axis=1) / (
        np.sqrt((eLc ** 2).sum(axis=1)) * np.sqrt((eRc ** 2).sum(axis=1)) + 1e-8
    )

    abs_diff = np.abs(eL - eR)
    prod = eL * eR

    feats = np.stack([
        cos,                               # 1
        angle,                             # 2
        dot,                               # 3
        l2,                                # 4
        l1,                                # 5
        nl2,                               # 6
        corr,                              # 7
        abs_diff.mean(axis=1),             # 8
        abs_diff.max(axis=1),              # 9
        abs_diff.std(axis=1),              # 10
        prod.mean(axis=1),                 # 11
        prod.max(axis=1),                  # 12
        prod.std(axis=1),                  # 13
    ], axis=1).astype(np.float32)
    return feats
