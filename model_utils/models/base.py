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
        
        # Debug: Print dropout configuration
        if dropout_rate > 0:
            print(f"[DEBUG] Dropout enabled with rate: {dropout_rate}")
        else:
            print(f"[DEBUG] Dropout disabled (rate: {dropout_rate})")

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
        
        # Debug: Print dropout status during training
        if self.training and self.dropout_rate > 0:
            print(f"[DEBUG] Dropout active during encoding (rate: {self.dropout_rate})")
        elif self.training and self.dropout_rate == 0:
            print(f"[DEBUG] Dropout inactive during training (rate: {self.dropout_rate})")
        elif not self.training and self.dropout_rate > 0:
            print(f"[DEBUG] Model in eval mode - dropout disabled (rate: {self.dropout_rate})")
        
        return F.normalize(z, dim=1)
    
    def to(self, device):
        """Move model to specified device."""
        super().to(device)
        if self.backbone:
            self.backbone.to(device)
        return self 