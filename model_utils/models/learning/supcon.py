from ..base import BaseSiameseModel
from utils.data import SupConDataset
from torch.utils.data import DataLoader
import torch

class SiameseModelSupCon(BaseSiameseModel):
    """
    Siamese network for SupCon learning using any vision-language model as backbone.
    Handles multiple positive and negative examples per anchor.
    """
    def __init__(self, embedding_dim=512, projection_dim=128, backbone=None):
        super().__init__(embedding_dim, projection_dim, backbone)
    
    def forward(self, anchor_text, positive_texts, negative_texts):
        """
        Forward pass for SupCon learning (batched).
        Args:
            anchor_text: list of anchor text strings, len=batch_size
            positive_texts: tuple or list of lists of positive text strings, shape [batch_size, n_positives] or [n_positives, batch_size]
            negative_texts: tuple or list of lists of negative text strings, shape [batch_size, n_negatives] or [n_negatives, batch_size]
        Returns:
            tuple: (z_anchor, z_positives, z_negatives) embeddings
        """
        # Encode anchors
        z_anchor = self.encode(anchor_text)  # [batch_size, emb_dim]
        
        # Handle positives - always stack then transpose to [batch_size, n_positives, emb_dim]
        z_positives = torch.stack([
            self.encode(pos_list) for pos_list in positive_texts
        ], dim=0).transpose(0, 1)  # [batch_size, n_positives, emb_dim]
        
        # Handle negatives - always stack then transpose to [batch_size, n_negatives, emb_dim]
        z_negatives = torch.stack([
            self.encode(neg_list) for neg_list in negative_texts
        ], dim=0).transpose(0, 1)  # [batch_size, n_negatives, emb_dim]
        
        return z_anchor, z_positives, z_negatives
