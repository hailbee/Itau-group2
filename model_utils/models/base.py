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
        if backbone is not None and tokenizer is not None:
            self.clip = backbone
            self.tokenizer = tokenizer
        else:
            self.clip = clip_model
            self.tokenizer = clip_tokenizer

        if freeze_clip and hasattr(self.clip, 'parameters'):
            for param in self.clip.parameters():
                param.requires_grad = False

        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def encode(self, texts):
        # Handle CLIPModelWrapper and SigLIPModelWrapper differently
        if hasattr(self.clip, 'encode_text'):
            # For CLIPModelWrapper and similar wrappers
            features = self.clip.encode_text(texts)
        elif hasattr(self.clip, 'processor') and hasattr(self.clip, 'get_text_features'):
            # For SigLIP-like models
            inputs = self.clip.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.projector[0].weight.device)
            with torch.no_grad():
                features = self.clip.get_text_features(**inputs)
        else:
            raise ValueError(f"Backbone {type(self.clip)} does not support text encoding")
        z = self.projector(features)
        return F.normalize(z, dim=1) 