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
            positive_texts: list of lists of positive text strings, shape [batch_size, n_positives]
            negative_texts: list of lists of negative text strings, shape [batch_size, n_negatives]
        Returns:
            tuple: (z_anchor, z_positives, z_negatives) embeddings
        """
        # Encode anchors
        z_anchor = self.encode(anchor_text)  # [batch_size, emb_dim]
        
        # Process positives: positive_texts is a list of lists [batch_size, n_positives]
        batch_size = len(anchor_text)
        n_positives = len(positive_texts[0]) if positive_texts else 0
        
        # Flatten all positives for batch encoding
        flat_positives = []
        for pos_list in positive_texts:
            flat_positives.extend(pos_list)
        
        # Encode all positives at once
        z_positives_flat = self.encode(flat_positives)  # [batch_size * n_positives, emb_dim]
        
        # Reshape to [batch_size, n_positives, emb_dim]
        z_positives = z_positives_flat.view(batch_size, n_positives, -1)
        
        # Process negatives: negative_texts is a list of lists [batch_size, n_negatives]
        n_negatives = len(negative_texts[0]) if negative_texts else 0
        
        # Flatten all negatives for batch encoding
        flat_negatives = []
        for neg_list in negative_texts:
            flat_negatives.extend(neg_list)
        
        # Encode all negatives at once
        z_negatives_flat = self.encode(flat_negatives)  # [batch_size * n_negatives, emb_dim]
        
        # Reshape to [batch_size, n_negatives, emb_dim]
        z_negatives = z_negatives_flat.view(batch_size, n_negatives, -1)
        
        return z_anchor, z_positives, z_negatives
