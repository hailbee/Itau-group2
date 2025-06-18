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
            negative_texts: list of list of negative text strings, shape [batch_size, n_negatives]
        Returns:
            tuple: (z_anchor, z_positive, z_negatives) embeddings
        """
        # Encode anchors
        z_anchor = self.encode(anchor_text)  # [batch_size, emb_dim]
        # Encode positives
        z_positive = self.encode(positive_text)  # [batch_size, emb_dim]
        # Encode negatives
        z_negatives = torch.stack([
            self.encode(neg_list) for neg_list in negative_texts
        ], dim=0)  # [batch_size, n_negatives, emb_dim]
        return z_anchor, z_positive, z_negatives

    @staticmethod
    def get_dataloader(dataframe, batch_size=256, num_workers=4):
        """
        Returns a DataLoader for InfoNCE training with fixed number of negatives.
        
        Args:
            dataframe: DataFrame containing the data
            batch_size: Batch size
            num_workers: Number of workers for data loading
            
        Returns:
            DataLoader: DataLoader for InfoNCE training
        """
        dataset = InfoNCEDataset(dataframe)
        return DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,  # No shuffling to maintain order
            num_workers=num_workers
        ) 