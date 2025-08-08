from ..base import BaseSiameseModel
from utils.data import InfoNCEDataset
from torch.utils.data import DataLoader
import torch

class SiameseModelInfoNCE(BaseSiameseModel):
    """
    Siamese network for InfoNCE learning using any vision-language model as backbone.
    Handles one positive and multiple negative examples per anchor.
    """
    def __init__(self, embedding_dim=512, projection_dim=128, backbone=None):
        super().__init__(embedding_dim, projection_dim, backbone)
    
    def forward(self, anchor_text, positive_text, negative_texts):
        """
        Forward pass for InfoNCE learning (batched).
        Args:
            anchor_text: list of anchor text strings, len=batch_size
            positive_text: list of positive text strings, len=batch_size
            negative_texts: list of lists of negative text strings, shape [batch_size, n_negatives]
        Returns:
            tuple: (z_anchor, z_positives, z_negatives) embeddings
        """
        # Encode anchors
        z_anchor = self.encode(anchor_text)  # [batch_size, emb_dim]
        # Encode positives
        z_positive = self.encode(positive_text)  # [batch_size, emb_dim]
        z_positives = z_positive.unsqueeze(1)  # [batch_size, 1, emb_dim]
        
        # Process negatives: negative_texts is a list of lists [batch_size, n_negatives]
        batch_size = len(anchor_text)
        n_negatives = len(negative_texts[0]) if negative_texts else 0
        
        # Debug: Check negative texts structure
        print(f"[DEBUG] InfoNCE forward: batch_size={batch_size}, n_negatives={n_negatives}")
        print(f"[DEBUG] InfoNCE forward: negative_texts type={type(negative_texts)}, length={len(negative_texts)}")
        if negative_texts and len(negative_texts) > 0:
            print(f"[DEBUG] InfoNCE forward: first negative list type={type(negative_texts[0])}, length={len(negative_texts[0])}")
        
        # Flatten all negatives for batch encoding
        flat_negatives = []
        for neg_list in negative_texts:
            flat_negatives.extend(neg_list)
        
        print(f"[DEBUG] InfoNCE forward: flat_negatives length={len(flat_negatives)}")
        
        # Encode all negatives at once
        z_negatives_flat = self.encode(flat_negatives)  # [batch_size * n_negatives, emb_dim]
        
        # Reshape to [batch_size, n_negatives, emb_dim]
        z_negatives = z_negatives_flat.view(batch_size, n_negatives, -1)
        
        return z_anchor, z_positives, z_negatives
