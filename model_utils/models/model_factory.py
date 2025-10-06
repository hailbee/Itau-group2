# model_utils/models/model_factory.py
from __future__ import annotations

from typing import Any, Optional

# Always available: our new precomputed backbone
from .wrappers.precomputed_wrapper import PrecomputedModelWrapper

# Optional imports for other backbones. If they are not present in your repo/env,
# the corresponding create_model branch will raise with a helpful message.
try:
    from .wrappers.clip_wrapper import CLIPModelWrapper  # type: ignore
except Exception:  # pragma: no cover
    CLIPModelWrapper = None  # type: ignore

try:
    from .wrappers.siglip_wrapper import SigLIPModelWrapper  # type: ignore
except Exception:  # pragma: no cover
    SigLIPModelWrapper = None  # type: ignore

try:
    from .wrappers.flava_wrapper import FLAVAModelWrapper  # type: ignore
except Exception:  # pragma: no cover
    FLAVAModelWrapper = None  # type: ignore

try:
    from .wrappers.coca_wrapper import CoCaModelWrapper  # type: ignore
except Exception:  # pragma: no cover
    CoCaModelWrapper = None  # type: ignore


class ModelFactory:
    """
    Central place to construct a "backbone wrapper" given a name.
    The returned object MUST expose:
      - .embedding_dim : int
      - .encode_text(List[str]) -> torch.FloatTensor [N, D]
    """

    @staticmethod
    def create_model(name: str, device: Optional[str] = None, **kwargs: Any):
        """
        name: backbone identifier (e.g., 'precomputed', 'clip', 'siglip', ...)
        device: optional, only relevant for learnable/torch models
        kwargs: backbone-specific arguments
        """
        if not name:
            raise ValueError("create_model: 'name' must be provided")
        key = name.lower().strip()

        if key == "precomputed":
            npz_paths = kwargs.get("npz_paths")
            if not npz_paths:
                raise ValueError("backbone='precomputed' requires npz_paths=[...]")
            return PrecomputedModelWrapper(
                npz_paths=npz_paths,
                lowercase_keys=kwargs.get("lowercase_keys", True),
                normalize=kwargs.get("normalize", False),
                strict=kwargs.get("strict", True),
            )

        if key == "clip":
            if CLIPModelWrapper is None:
                raise ImportError("CLIPModelWrapper not available. Ensure wrappers/clip_wrapper.py and deps are installed.")
            return CLIPModelWrapper(model_name=kwargs.get("model_name"), device=device)

        if key == "siglip":
            if SigLIPModelWrapper is None:
                raise ImportError("SigLIPModelWrapper not available. Ensure wrappers/siglip_wrapper.py and deps are installed.")
            return SigLIPModelWrapper(model_name=kwargs.get("model_name"), device=device)

        if key == "flava":
            if FLAVAModelWrapper is None:
                raise ImportError("FLAVAModelWrapper not available. Ensure wrappers/flava_wrapper.py and deps are installed.")
            return FLAVAModelWrapper(model_name=kwargs.get("model_name"), device=device)

        if key == "coca":
            if CoCaModelWrapper is None:
                raise ImportError("CoCaModelWrapper not available. Ensure wrappers/coca_wrapper.py and deps are installed.")
            return CoCaModelWrapper(model_name=kwargs.get("model_name"), device=device)

        raise ValueError(f"Unknown backbone: {name!r}")