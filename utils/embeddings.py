import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

# Load CLIP model and tokenizer
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

### Retrieving embeddings from trained model
class EmbeddingExtractor(nn.Module):
    def __init__(self, siamese_model):
        super().__init__()
        self.clip = siamese_model.clip
        self.projector = siamese_model.projector

    def forward(self, texts):
        inputs = clip_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.projector[0].weight.device)
        with torch.no_grad():
            features = self.clip.get_text_features(**inputs)
        projected = self.projector(features)
        return F.normalize(projected, dim=1)

def batched_embedding(extractor, names, batch_size=32):
    embeddings = []
    for i in range(0, len(names), batch_size):
        batch = names[i:i+batch_size]
        emb = extractor(batch)
        embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)

def get_clip_embeddings(texts, batch_size=32):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = clip_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(clip_model.device)
        outputs = clip_model.get_text_features(**inputs)
        normed = F.normalize(outputs, dim=1)
        all_embeddings.append(normed)
    return torch.cat(all_embeddings, dim=0)
