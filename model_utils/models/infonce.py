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
        Forward pass for InfoNCE learning.
        
        Args:
            anchor_text: Anchor text input
            positive_text: Single positive text input
            negative_texts: List of negative text inputs (exactly 3)
            
        Returns:
            tuple: (z_anchor, z_positive, z_negatives) embeddings
        """
        # Encode anchor
        z_anchor = self.encode(anchor_text)
        
        # Encode positive
        z_positive = self.encode(positive_text)
        
        # Encode negatives and stack them
        z_negatives = torch.stack([self.encode(neg) for neg in negative_texts], dim=1)
        
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