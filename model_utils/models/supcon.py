from .base import BaseSiameseCLIP
from utils.data import ClassBalancedBatchSampler, TextPairDataset
from torch.utils.data import DataLoader
import pandas as pd
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
            positive_texts: List of positive text inputs
            negative_texts: List of negative text inputs
            
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
    def get_dataloader(dataframe, batch_size=256, n_positives=3, n_negatives=3):
        """
        Returns a DataLoader with class-balanced sampling for SupCon.
        Ensures >= n_positives per class per batch.
        
        Args:
            dataframe: DataFrame containing the data
            batch_size: Batch size
            n_positives: Number of positive examples per anchor
            n_negatives: Number of negative examples per anchor
            
        Returns:
            DataLoader: DataLoader for SupCon training
        """
        dataset = TextPairDataset(dataframe)
        labels = dataframe['label'].tolist()
        sampler = ClassBalancedBatchSampler(labels, batch_size, n_positives=n_positives)
        return DataLoader(dataset, batch_sampler=sampler) 