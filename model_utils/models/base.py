import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseSiameseModel(nn.Module):
    """
    Base class for siamese models that can work with any vision-language model.
    """
    def __init__(self, embedding_dim=512, projection_dim=128, backbone=None, dropout_rate=0.0):
        super().__init__()
        self.backbone = backbone  # Model wrapper (CLIP, FLAVA, etc.)
        self.dropout_rate = dropout_rate
        
        # Add dropout layers to the projector
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(projection_dim, projection_dim),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

    def get_dropout_info(self):
        """Get information about dropout configuration."""
        dropout_layers = [layer for layer in self.projector if isinstance(layer, nn.Dropout)]
        return {
            'dropout_rate': self.dropout_rate,
            'num_dropout_layers': len(dropout_layers),
            'dropout_layers': dropout_layers
        }

    def encode(self, texts):
        """
        Encode texts using the backbone model and project to embedding space.
        
        Args:
            texts: List of text strings to encode
            
        Returns:
            Normalized embeddings
        """
        features = self.backbone.encode_text(texts)
        z = self.projector(features)
        
        return F.normalize(z, dim=1)
    
    def to(self, device):
        """Move model to specified device."""
        super().to(device)
        if self.backbone:
            self.backbone.to(device)
        return self 