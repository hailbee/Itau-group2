import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, SiglipTextModel

# Check if sentencepiece is available
try:
    import sentencepiece
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False

class SigLIPModelWrapper:
    """Wrapper for SigLIP text-only models."""
    
    def __init__(self, model_name="google/siglip-base-patch16-224", device=None):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        if not SENTENCEPIECE_AVAILABLE:
            raise ImportError(
                "SigLIP models require the 'sentencepiece' library. "
                "Please install it with: pip install sentencepiece==0.2.0"
            )
        self.model = SiglipTextModel.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float32
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )
    
    def encode_text(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Always pool to [batch_size, hidden_size]
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                features = outputs.pooler_output  # [batch_size, hidden_size]
            elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
                features = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
            else:
                # fallback: find first 2D tensor in outputs
                features = None
                if isinstance(outputs, dict):
                    for v in outputs.values():
                        if v is not None and len(v.shape) == 2:
                            features = v
                            break
                if features is None:
                    raise ValueError(f"Could not extract 2D features from SigLIP model output: {type(outputs)}; output: {outputs}")
        return F.normalize(features, dim=1)

    @property
    def embedding_dim(self):
        # SigLIP: hidden_size for text encoder
        return self.model.config.hidden_size
    
    def to(self, device):
        """Move model to specified device."""
        self.device = device
        if self.model:
            self.model = self.model.to(device)
        return self 