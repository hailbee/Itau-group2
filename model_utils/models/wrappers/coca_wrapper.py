import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class CoCaModelWrapper:
    """Wrapper for CoCa/GIT models."""
    
    def __init__(self, model_name="microsoft/git-base-coco", device=None):
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
            # GIT/CoCa models use the last hidden state for text features
            if hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state.mean(dim=1)  # Pool over sequence length
            elif hasattr(outputs, 'text_embeds'):
                features = outputs.text_embeds
            else:
                # Fallback to logits if available
                features = outputs.logits.mean(dim=1)
        return F.normalize(features, dim=1)

    @property
    def embedding_dim(self):
        # GIT/CoCa models: usually hidden_size
        return self.model.config.hidden_size
    
    def to(self, device):
        """Move model to specified device."""
        self.device = device
        if self.model:
            self.model = self.model.to(device)
        return self 