import torch
from torch.utils.data import Dataset
import numpy as np

class TextPairDataset(Dataset):
    def __init__(self, dataframe):
        self.name1 = dataframe['name1'].tolist()
        self.name2 = dataframe['name2'].tolist()
        self.label = dataframe['label'].tolist()

    def __len__(self):
        return len(self.name1)

    def __getitem__(self, idx):
        return self.name1[idx], self.name2[idx], torch.tensor(self.label[idx], dtype=torch.float32)

class TripletDataset(Dataset):
    def __init__(self, dataframe):
        self.anchor_data = dataframe['fraud_name'].tolist()
        self.positive_data = dataframe['real_name'].tolist()
        self.negative_data = dataframe['negative_name'].tolist()

    def __len__(self):
        return len(self.anchor_data)

    def __getitem__(self, idx):
        return self.anchor_data[idx], self.positive_data[idx], self.negative_data[idx]

class SupConDataset(Dataset):
    def __init__(self, dataframe):
        """
        Dataset for SupCon that handles multiple positives and negatives per anchor.
        Each anchor should have exactly 3 positives and 3 negatives.
        
        Args:
            dataframe: DataFrame with columns 'fraud_name' (anchor), 'real_name' (list of positives),
                      'negative_name' (list of negatives)
        """
        self.anchor_data = dataframe['real_name'].tolist()
        self.positive_data = dataframe['positive_names'].tolist()  # Should be list of lists
        self.negative_data = dataframe['negative_names'].tolist()  # Should be list of lists

    def __len__(self):
        return len(self.anchor_data)

    def __getitem__(self, idx):
        """Returns anchor, exactly 3 positives and 3 negatives"""
        anchor = self.anchor_data[idx]
        positives = self.positive_data[idx]
        negatives = self.negative_data[idx][:7]
        
        # Pad negatives
        if len(negatives) < 7:
            negatives = negatives + [negatives[0]] * (7 - len(negatives))
            
        return anchor, positives, negatives

class InfoNCEDataset(Dataset):
    def __init__(self, dataframe):
        """
        Dataset for InfoNCE that handles one positive and multiple negatives per anchor.
        Each anchor should have exactly 1 positive and 3 negatives.
        
        Args:
            dataframe: DataFrame with columns 'fraud_name' (anchor), 'real_name' (positive),
                      'negative_name' (list of negatives)
        """
        self.anchor_data = dataframe['anchor_name'].tolist()
        self.positive_data = dataframe['positive_name'].tolist()
        self.negative_data = dataframe['negative_names'].tolist()  # Should be list of lists

    def __len__(self):
        return len(self.anchor_data)

    def __getitem__(self, idx):
        """Returns anchor, one positive, and exactly 3 negatives"""
        anchor = self.anchor_data[idx]
        positive = self.positive_data[idx]
        negatives = self.negative_data[idx][:3]  # Take first 3 negatives
        
        # Pad if necessary
        if len(negatives) < 3:
            negatives = negatives + [negatives[0]] * (3 - len(negatives))
            
        return anchor, positive, negatives
