# model_utils/models/base.py
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseSiameseModel(nn.Module):
    """
    Thin model that wraps a backbone text-encoder wrapper and (optionally) a projector head.
    Requirements on `backbone`:
      - .embedding_dim : int
      - .encode_text(List[str]) -> torch.FloatTensor [N, D]
    """

    def __init__(self, backbone, projection_dim: int = 128, use_projector: bool = True):
        super().__init__()
        self.backbone = backbone
        self.use_projector = use_projector

        in_dim = getattr(backbone, "embedding_dim", None)
        if in_dim is None:
            raise ValueError("Backbone must expose .embedding_dim")

        if use_projector:
            # Lightweight 2-layer MLP projector
            self.projector = nn.Sequential(
                nn.Linear(in_dim, projection_dim),
                nn.ReLU(inplace=True),
                nn.Linear(projection_dim, projection_dim),
            )
            self.out_dim = projection_dim
        else:
            # Identity: output stays in backbone's embedding space
            self.projector = nn.Identity()
            self.out_dim = in_dim

    @torch.inference_mode()
    def encode(self, texts: List[str]) -> torch.FloatTensor:
        """
        Returns L2-normalized embeddings. If backbone returns already-normalized
        vectors (e.g., precomputed), the extra normalization is harmless.
        """
        # Backbone handles any device specifics internally (or returns CPU tensors)
        z = self.backbone.encode_text(texts)  # [N, D]
        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.float32)
        z = self.projector(z)                 # [N, P] or Identity
        z = F.normalize(z, dim=1)
        return z
