import torch
from torch.utils.data import Dataset
import numpy as np
import ast

class TextPairDataset(Dataset):
    def __init__(self, dataframe):
        # Handle both old format (name1, name2, label) and new format (fraudulent_name, real_name, label)
        if 'fraudulent_name' in dataframe.columns and 'real_name' in dataframe.columns:
            self.name1 = dataframe['fraudulent_name'].tolist()
            self.name2 = dataframe['real_name'].tolist()
        elif 'name1' in dataframe.columns and 'name2' in dataframe.columns:
            self.name1 = dataframe['name1'].tolist()
            self.name2 = dataframe['name2'].tolist()
        else:
            raise ValueError("DataFrame must have either (fraudulent_name, real_name) or (name1, name2) columns")
        
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
    def __init__(self, dataframe, max_positives=3, max_negatives=7):
        """
        Dataset for SupCon that handles multiple positives and negatives per anchor.
        Automatically adapts to consistent or inconsistent positive/negative counts.
        
        Args:
            dataframe: DataFrame with columns ['anchor_name', 'positive_names', 'negative_names']
            max_positives: Maximum number of positives to use per sample (default: 3)
            max_negatives: Maximum number of negatives to use per sample (default: 7)
        """
        self.anchor_data = dataframe['anchor_name'].tolist()
        
        # Handle positive_names - they might be string representations of lists
        raw_positive_data = dataframe['positive_names'].tolist()
        self.positive_data = []
        for pos_item in raw_positive_data:
            if isinstance(pos_item, str):
                # Parse string representation of list
                try:
                    pos_list = ast.literal_eval(pos_item)
                    if isinstance(pos_list, list):
                        self.positive_data.append(pos_list)
                    else:
                        # If it's not a list, treat as single item
                        self.positive_data.append([pos_item])
                except (ValueError, SyntaxError):
                    # If parsing fails, treat as single item
                    self.positive_data.append([pos_item])
            else:
                # Already a list
                self.positive_data.append(pos_item)
        
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
        
        # Check if all samples have the same number of positives and negatives
        pos_lengths = [len(pos_list) for pos_list in self.positive_data]
        neg_lengths = [len(neg_list) for neg_list in self.negative_data]
        unique_pos_lengths = set(pos_lengths)
        unique_neg_lengths = set(neg_lengths)
        
        if len(unique_pos_lengths) == 1 and len(unique_neg_lengths) == 1:
            # All samples have the same number of positives and negatives
            self.consistent_positives = True
            self.consistent_negatives = True
            self.n_positives = list(unique_pos_lengths)[0]
            self.n_negatives = list(unique_neg_lengths)[0]
            print(f"SupConDataset: Consistent counts detected. All samples have {self.n_positives} positives and {self.n_negatives} negatives.")
        else:
            # Inconsistent counts
            self.consistent_positives = len(unique_pos_lengths) == 1
            self.consistent_negatives = len(unique_neg_lengths) == 1
            self.n_positives = max_positives
            self.n_negatives = max_negatives
            print(f"SupConDataset: Inconsistent counts detected. Using max_positives={max_positives}, max_negatives={max_negatives}.")
            print(f"Positive count range: {min(unique_pos_lengths)} to {max(unique_pos_lengths)}")
            print(f"Negative count range: {min(unique_neg_lengths)} to {max(unique_neg_lengths)}")

    def __len__(self):
        return len(self.anchor_data)

    def __getitem__(self, idx):
        """Returns anchor, positives, and negatives (adapted to dataset type)"""
        anchor = self.anchor_data[idx]
        positives = self.positive_data[idx]
        negatives = self.negative_data[idx]
        
        # Handle positives
        if self.consistent_positives:
            # Use all positives as-is
            pass
        else:
            # Pad or truncate to max_positives
            if len(positives) >= self.n_positives:
                positives = positives[:self.n_positives]  # Truncate
            else:
                # Pad by repeating the first positive
                positives = positives + [positives[0]] * (self.n_positives - len(positives))
        
        # Handle negatives
        if self.consistent_negatives:
            # Use all negatives as-is
            pass
        else:
            # Pad or truncate to max_negatives
            if len(negatives) >= self.n_negatives:
                negatives = negatives[:self.n_negatives]  # Truncate
            else:
                # Pad by repeating the first negative
                negatives = negatives + [negatives[0]] * (self.n_negatives - len(negatives))
            
        return anchor, positives, negatives

def infonce_collate_fn(batch):
    anchors, positives, negatives = zip(*batch)
    return list(anchors), list(positives), list(negatives)

def supcon_collate_fn(batch):
    anchors, positives, negatives = zip(*batch)
    return list(anchors), list(positives), list(negatives)

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
