import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

class CLIPModelWrapper:
    """Wrapper for OpenAI CLIP models."""
    
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
    
    def encode_text(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return F.normalize(features, dim=1)

    @property
    def embedding_dim(self):
        # CLIP text encoder output dim
        return self.model.text_projection.out_features
    
    def to(self, device):
        """Move model to specified device."""
        self.device = device
        if self.model:
            self.model = self.model.to(device)
        return self 