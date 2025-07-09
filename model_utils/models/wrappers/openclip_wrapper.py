import torch
import torch.nn.functional as F

class OpenCLIPModelWrapper:
    """Wrapper for OpenCLIP models."""
    
    def __init__(self, model_name="ViT-L-14", device=None):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained='openai'
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model = self.model.to(self.device)
    
    def encode_text(self, texts):
        with torch.no_grad():
            text_tokens = self.tokenizer(texts).to(self.device)
            features = self.model.encode_text(text_tokens)
        return F.normalize(features, dim=1)

    @property
    def embedding_dim(self):
        # OpenCLIP: text_projection shape
        return self.model.text_projection.weight.shape[0]
    
    def to(self, device):
        """Move model to specified device."""
        self.device = device
        if self.model:
            self.model = self.model.to(device)
        return self 