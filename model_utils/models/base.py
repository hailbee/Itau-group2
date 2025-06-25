import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

class BaseSiameseCLIP(nn.Module):
    def __init__(self, embedding_dim=512, projection_dim=128, freeze_clip=True):
        super().__init__()
        # Load CLIP model and tokenizer
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False

        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def to(self, device):
        """Override to method to ensure CLIP model is also moved to device"""
        super().to(device)
        self.clip = self.clip.to(device)
        return self

    def encode(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.clip.device)
        with torch.no_grad():
            features = self.clip.get_text_features(**inputs)
        z = self.projector(features)
        return F.normalize(z, dim=1) 