from .base import BaseSiameseCLIP
from utils.data import ClassBalancedBatchSampler, TextPairDataset
from torch.utils.data import DataLoader
import pandas as pd
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
            negative_texts: List of negative text inputs
            
        Returns:
            tuple: (z_anchor, z_positive, z_negatives) embeddings
        """
        # Encode anchor and positive
        z_anchor = self.encode(anchor_text)
        z_positive = self.encode(positive_text)
        
        # Encode negatives and stack them
        z_negatives = torch.stack([self.encode(neg) for neg in negative_texts], dim=1)
        
        return z_anchor, z_positive, z_negatives

    @staticmethod
    def get_dataloader(dataframe, batch_size=256, n_negatives=3):
        """
        Returns a DataLoader with class-balanced sampling for InfoNCE.
        Ensures one positive and n_negatives per anchor.
        
        Args:
            dataframe: DataFrame containing the data
            batch_size: Batch size
            n_negatives: Number of negative examples per anchor
            
        Returns:
            DataLoader: DataLoader for InfoNCE training
        """
        dataset = TextPairDataset(dataframe)
        labels = dataframe['label'].tolist()
        sampler = ClassBalancedBatchSampler(labels, batch_size, n_positives=1)  # Only need 1 positive for InfoNCE
        return DataLoader(dataset, batch_sampler=sampler)
