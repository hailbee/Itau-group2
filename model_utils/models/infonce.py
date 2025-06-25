from .base import BaseSiameseCLIP
from utils.data import InfoNCEDataset
from torch.utils.data import DataLoader
import torch

class SiameseCLIPInfoNCE(BaseSiameseCLIP):
    """
    Siamese network for InfoNCE learning using CLIP as backbone.
    Handles one positive and multiple negative examples per anchor.
    """
    def forward(self, anchor_text, positive_text, negative_texts):
        """
        Forward pass for InfoNCE learning (batched).
        Args:
            anchor_text: list of anchor text strings, len=batch_size
            positive_text: list of positive text strings, len=batch_size
            negative_texts: tuple of lists of negative text strings, shape [batch_size, n_negatives]
        Returns:
            tuple: (z_anchor, z_positives, z_negatives) embeddings
        """
        # Debug prints
        print(f"DEBUG InfoNCE forward:")
        print(f"  anchor_text type: {type(anchor_text)}, length: {len(anchor_text)}")
        print(f"  positive_text type: {type(positive_text)}, length: {len(positive_text)}")
        print(f"  negative_texts type: {type(negative_texts)}, length: {len(negative_texts)}")
        print(f"  negative_texts[0] type: {type(negative_texts[0])}, length: {len(negative_texts[0])}")
        print(f"  negative_texts[0][0]: {negative_texts[0][0]}")
        
        # Encode anchors
        z_anchor = self.encode(anchor_text)  # [batch_size, emb_dim]
        # Encode positives
        z_positive = self.encode(positive_text)  # [batch_size, emb_dim]
        z_positives = z_positive.unsqueeze(1)  # [batch_size, 1, emb_dim]
        # Encode negatives
        # negative_texts is a tuple of lists: ([neg1_sample1, neg2_sample1, ...], [neg1_sample2, neg2_sample2, ...], ...)
        batch_size = len(anchor_text)
        n_negatives = len(negative_texts[0])  # All samples should have same number of negatives
        
        print(f"  batch_size: {batch_size}, n_negatives: {n_negatives}")
        
        # Flatten the negative texts for batch encoding
        flat_negatives = []
        for sample_negatives in negative_texts:
            flat_negatives.extend(sample_negatives)
        
        print(f"  flat_negatives length: {len(flat_negatives)}")
        print(f"  expected length: {batch_size * n_negatives}")
        
        z_negatives_flat = self.encode(flat_negatives)  # [batch_size * n_negatives, emb_dim]
        print(f"  z_negatives_flat shape: {z_negatives_flat.shape}")
        
        z_negatives = z_negatives_flat.view(batch_size, n_negatives, -1)  # [batch_size, n_negatives, emb_dim]
        print(f"  z_negatives shape: {z_negatives.shape}")
        
        return z_anchor, z_positives, z_negatives
