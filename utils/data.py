import torch
from torch.utils.data import Dataset
import numpy as np
import ast

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
        Each anchor should have exactly 3 positives and 7 negatives.
        
        Args:
            dataframe: DataFrame with columns 'real_name' (anchor), 'positive_names' (list of positives),
                      'negative_names' (list of negatives)
        """
        self.anchor_data = dataframe['anchor_name'].tolist()
        self.positive_data = dataframe['positive_names'].tolist()  # Should be list of lists
        self.negative_data = dataframe['negative_names'].tolist()  # Should be list of lists

    def __len__(self):
        return len(self.anchor_data)

    def __getitem__(self, idx):
        """Returns anchor, exactly 3 positives and 7 negatives"""
        anchor = self.anchor_data[idx]
        positives = self.positive_data[idx]
        negatives = self.negative_data[idx][:7]  # Take first 7 negatives
        
        # Pad negatives if necessary
        if len(negatives) < 7:
            negatives = negatives + [negatives[0]] * (7 - len(negatives))
            
        return anchor, positives, negatives

class InfoNCEDataset(Dataset):
    def __init__(self, dataframe, max_negatives=6):
        """
        Dataset for InfoNCE that handles one positive and multiple negatives per anchor.
        Automatically adapts to consistent or inconsistent negative counts.
        
        Args:
            dataframe: DataFrame with columns 'anchor_name' (anchor), 'positive_name' (positive),
                      'negative_names' (list of negatives)
            max_negatives: Maximum number of negatives to use per sample (default: 6)
        """
        self.anchor_data = dataframe['anchor_name'].tolist()
        self.positive_data = dataframe['positive_name'].tolist()
        
        # Handle negative_names - they might be string representations of lists
        raw_negative_data = dataframe['negative_names'].tolist()
        self.negative_data = []
        for neg_item in raw_negative_data:
            if isinstance(neg_item, str):
                # Parse string representation of list
                try:
                    neg_list = ast.literal_eval(neg_item)
                    if isinstance(neg_list, list):
                        self.negative_data.append(neg_list)
                    else:
                        # If it's not a list, treat as single item
                        self.negative_data.append([neg_item])
                except (ValueError, SyntaxError):
                    # If parsing fails, treat as single item
                    self.negative_data.append([neg_item])
            else:
                # Already a list
                self.negative_data.append(neg_item)
        
        # Check if all samples have the same number of negatives
        neg_lengths = [len(neg_list) for neg_list in self.negative_data]
        unique_lengths = set(neg_lengths)
        
        if len(unique_lengths) == 1:
            # All samples have the same number of negatives
            self.consistent_negatives = True
            self.n_negatives = list(unique_lengths)[0]
            print(f"InfoNCEDataset: Consistent negative count detected. All samples have {self.n_negatives} negatives.")
        else:
            # Inconsistent negative counts
            self.consistent_negatives = False
            self.n_negatives = max_negatives
            print(f"InfoNCEDataset: Inconsistent negative counts detected. Using max_negatives={max_negatives}.")
            print(f"Negative count range: {min(unique_lengths)} to {max(unique_lengths)}")

    def __len__(self):
        return len(self.anchor_data)

    def __getitem__(self, idx):
        """Returns anchor, one positive, and negatives (adapted to dataset type)"""
        anchor = self.anchor_data[idx]
        positive = self.positive_data[idx]
        negatives = self.negative_data[idx]
        
        if self.consistent_negatives:
            # Use all negatives as-is
            return anchor, positive, negatives
        else:
            # Pad or truncate to max_negatives
            if len(negatives) >= self.n_negatives:
                negatives = negatives[:self.n_negatives]  # Truncate
            else:
                # Pad by repeating the first negative
                negatives = negatives + [negatives[0]] * (self.n_negatives - len(negatives))
            
            return anchor, positive, negatives
