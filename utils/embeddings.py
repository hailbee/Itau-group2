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
        self.siamese_model = siamese_model

    def forward(self, texts):
        # Use the siamese model's encode method which handles the backbone and projection
        return self.siamese_model.encode(texts)
    
    def encode(self, texts):
        # Alias for forward method to maintain compatibility
        return self.forward(texts)

class SupConEmbeddingExtractor(nn.Module):
    """
    Specialized embedding extractor for SupCon models that can handle single text input for evaluation.
    """
    def __init__(self, siamese_model):
        super().__init__()
        self.siamese_model = siamese_model

    def forward(self, texts):
        # For evaluation, we just need to encode the texts normally
        # Use the siamese model's encode method which handles the backbone and projection
        return self.siamese_model.encode(texts)
    
    def encode(self, texts):
        # Alias for forward method to maintain compatibility
        return self.forward(texts)

def batched_embedding(extractor, names, batch_size=32):
    embeddings = []
    for i in range(0, len(names), batch_size):
        batch = names[i:i+batch_size]
        emb = extractor(batch)
        # Move to CPU for concatenation
        if hasattr(emb, 'cpu'):
            embeddings.append(emb.cpu())
        else:
            embeddings.append(emb)
    return torch.cat(embeddings, dim=0)

