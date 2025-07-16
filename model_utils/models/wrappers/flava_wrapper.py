import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class FLAVAModelWrapper:
    """Wrapper for FLAVA models."""
    
    def __init__(self, model_name="facebook/flava-full", device=None):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def encode_text(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # FLAVA models have different output structures
            if hasattr(outputs, 'text_embeds'):
                features = outputs.text_embeds
            elif hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state.mean(dim=1)
            elif hasattr(outputs, 'pooler_output'):
                features = outputs.pooler_output
            elif hasattr(outputs, 'text_outputs') and hasattr(outputs.text_outputs, 'last_hidden_state'):
                features = outputs.text_outputs.last_hidden_state.mean(dim=1)
            else:
                # Try to get any available tensor output
                for attr in dir(outputs):
                    if not attr.startswith('_') and hasattr(getattr(outputs, attr), 'shape'):
                        tensor = getattr(outputs, attr)
                        if len(tensor.shape) >= 2:
                            features = tensor.mean(dim=1) if len(tensor.shape) > 2 else tensor
                            break
                else:
                    raise ValueError(f"Could not extract features from FLAVA model output: {type(outputs)}")
        return F.normalize(features, dim=1)

    @property
    def embedding_dim(self):
        # FLAVA: usually hidden_size
        return self.model.config.hidden_size
    
    def to(self, device):
        """Move model to specified device."""
        self.device = device
        if self.model:
            self.model = self.model.to(device)
        return self 