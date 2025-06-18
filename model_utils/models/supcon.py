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
        Forward pass for SupCon learning.
        
        Args:
            anchor_text: Anchor text input
            positive_texts: List of positive text inputs (exactly 3)
            negative_texts: List of negative text inputs (exactly 3)
            
        Returns:
            tuple: (z_anchor, z_positives, z_negatives) embeddings
        """
        # Encode anchor
        z_anchor = self.encode(anchor_text)
        
        # Encode positives and stack them
        z_positives = torch.stack([self.encode(pos) for pos in positive_texts], dim=1)
        
        # Encode negatives and stack them
        z_negatives = torch.stack([self.encode(neg) for neg in negative_texts], dim=1)
        
        return z_anchor, z_positives, z_negatives

    @staticmethod
    def get_dataloader(dataframe, batch_size=256, num_workers=4):
        """
        Returns a DataLoader for SupCon training with fixed numbers of positives and negatives.
        
        Args:
            dataframe: DataFrame containing the data
            batch_size: Batch size
            num_workers: Number of workers for data loading
            
        Returns:
            DataLoader: DataLoader for SupCon training
        """
        dataset = SupConDataset(dataframe)
        return DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,  # No shuffling to maintain order
            num_workers=num_workers
        ) 