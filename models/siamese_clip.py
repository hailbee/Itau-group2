import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

# Load CLIP model and tokenizer
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Siamese Model using CLIP
class SiameseCLIPModel(nn.Module):
    def __init__(self, embedding_dim=512, projection_dim=128, freeze_clip=True):
        super(SiameseCLIPModel, self).__init__()
        self.clip = clip_model

        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False

        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def encode(self, texts):
        inputs = clip_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.clip.device)
        with torch.no_grad():
            features = self.clip.get_text_features(**inputs)
        z = self.projector(features)
        return F.normalize(z, dim=1)

    def forward(self, text1, text2):
        z1 = self.encode(text1)
        z2 = self.encode(text2)
        return z1, z2