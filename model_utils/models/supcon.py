from .base import BaseSiameseCLIP
from utils.data import SupConDataset
from torch.utils.data import DataLoader
import torch

class SiameseCLIPSupCon(BaseSiameseCLIP):
    """
    Siamese network for SupCon learning using CLIP as backbone.
    Handles multiple positive and negative examples per anchor.
    """
    def forward(self, anchor_text, positive_texts, negative_texts):
        """
        Forward pass for SupCon learning (batched).
        Args:
            anchor_text: list of anchor text strings, len=batch_size
            positive_texts: tuple of lists of positive text strings, shape [batch_size, n_positives]
            negative_texts: tuple of lists of negative text strings, shape [batch_size, n_negatives]
        Returns:
            tuple: (z_anchor, z_positives, z_negatives) embeddings
        """
        # Encode anchors
        z_anchor = self.encode(anchor_text)  # [batch_size, emb_dim]
        
        # Handle positives - transpose if needed
        if isinstance(positive_texts, tuple) and isinstance(positive_texts[0], list):
            # positive_texts is tuple of lists: (pos1_batch, pos2_batch, pos3_batch, ...)
            batch_size = len(anchor_text)
            n_positives = len(positive_texts)
            
            # Flatten the positive texts for batch encoding
            flat_positives = []
            for pos_batch in positive_texts:  # pos_batch is a list of strings for this positive position
                flat_positives.extend(pos_batch)
            
            z_positives_flat = self.encode(flat_positives)  # [batch_size * n_positives, emb_dim]
            z_positives = z_positives_flat.view(n_positives, batch_size, -1).transpose(0, 1)  # [batch_size, n_positives, emb_dim]
        else:
            # positive_texts is already in the expected format
            z_positives = torch.stack([
                self.encode(pos_list) for pos_list in positive_texts
            ], dim=0)  # [batch_size, n_positives, emb_dim]
        
        # Handle negatives - transpose if needed
        if isinstance(negative_texts, tuple) and isinstance(negative_texts[0], list):
            # negative_texts is tuple of lists: (neg1_batch, neg2_batch, neg3_batch, ...)
            batch_size = len(anchor_text)
            n_negatives = len(negative_texts)
            
            # Flatten the negative texts for batch encoding
            flat_negatives = []
            for neg_batch in negative_texts:  # neg_batch is a list of strings for this negative position
                flat_negatives.extend(neg_batch)
            
            z_negatives_flat = self.encode(flat_negatives)  # [batch_size * n_negatives, emb_dim]
            z_negatives = z_negatives_flat.view(n_negatives, batch_size, -1).transpose(0, 1)  # [batch_size, n_negatives, emb_dim]
        else:
            # negative_texts is already in the expected format
            z_negatives = torch.stack([
                self.encode(neg_list) for neg_list in negative_texts
            ], dim=0)  # [batch_size, n_negatives, emb_dim]
        
        return z_anchor, z_positives, z_negatives
