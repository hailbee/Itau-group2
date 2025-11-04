from typing import Dict, List, Sequence, Optional
import numpy as np
import torch
from tqdm import tqdm
import os

class PrecomputedModelWrapper:
    """
    Backbone wrapper that serves precomputed word embeddings from one or more .npz files.
    Each NPZ must map `word -> 1D vector` (already L2-normalized, unless `normalize=True`).
    """
    def __init__(
        self,
        npz_paths: Sequence[str],
        lowercase_keys: bool = True,
        normalize: bool = False,
        strict: bool = True,
    ):
        self.lowercase_keys = lowercase_keys
        self.normalize = normalize
        self.strict = strict
        self._store: Dict[str, np.ndarray] = {}

        for p in npz_paths:
            p = os.path.abspath(p)
            data = np.load(p, allow_pickle=True)
            for k in tqdm(data.files):
                key = k.split(".")[0]
                key = key.lower() if lowercase_keys else key
                v = data[k].astype(np.float32)
                if normalize:
                    n = float(np.linalg.norm(v))
                    if n > 0:
                        v = v / n
                self._store[key] = v

        if not self._store:
            raise ValueError("No embeddings loaded for precomputed backbone.")
        
        # After building self._store from NPZs:
        first = next(iter(self._store.values()))
        first = np.asarray(first).squeeze()
        self.embedding_dim = int(first.reshape(-1).shape[0])

        # Coerce all to 1D consistently
        for k in list(self._store.keys()):
            self._store[k] = self._coerce_1d(self._store[k])

        # self.embedding_dim = int(next(iter(self._store.values())).shape[-1])
        self.is_precomputed = True  # marker used by BaseSiameseModel/main if needed

    def _coerce_1d(self, v: np.ndarray) -> np.ndarray:
        # Ensure a flat 1D vector of length embedding_dim
        v = np.asarray(v, dtype=np.float32).squeeze()
        if v.ndim > 1:
            v = v.reshape(-1)
        if v.size != self.embedding_dim:
            raise ValueError(f"Embedding has size {v.size}, expected {self.embedding_dim}")
        if self.normalize:
            n = float(np.linalg.norm(v))
            if n > 0:
                v = v / n
        return v

    def _lookup(self, word: str) -> Optional[np.ndarray]:
        key = word if self.lowercase_keys else word
        if self.lowercase_keys:
            key = word.lower()
        vec = self._store.get(key)
        if vec is None and self.strict:
            raise KeyError(f"Missing embedding for word: {word!r}")
        return vec

    def encode_text(self, texts: List[str]) -> torch.FloatTensor:
        outs = []
        for t in texts:
            key = t if self.lowercase_keys else t
            if self.lowercase_keys:
                key = t.lower()
            v = self._store.get(key)
            if v is None:
                if self.strict:
                    raise KeyError(f"Missing embedding for word: {t!r}")
                v = np.zeros(self.embedding_dim, dtype=np.float32)
            else:
                v = self._coerce_1d(v)  # <-- ensures shape (D,)
            outs.append(torch.from_numpy(v).unsqueeze(0))  # [1, D]
        return torch.cat(outs, dim=0)  # [N, D]

