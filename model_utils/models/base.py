import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

# Load CLIP model and tokenizer
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

class BaseSiameseCLIP(nn.Module):
    def __init__(self, embedding_dim=512, projection_dim=128, freeze_clip=True, backbone=None, tokenizer=None):
        super().__init__()
        self.backbone = backbone  # Always the wrapper
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def encode(self, texts):
        features = self.backbone.encode_text(texts)
        z = self.projector(features)
        return F.normalize(z, dim=1) 