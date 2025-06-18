import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
from collections import defaultdict

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

class ClassBalancedBatchSampler(Sampler):
    """
    Samples batches with at least n_positives per class (for SupCon) or one positive per anchor (for InfoNCE).
    """
    def __init__(self, labels, batch_size, n_positives=2):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.n_positives = n_positives
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.class_indices[label].append(idx)
        self.classes = list(self.class_indices.keys())

    def __iter__(self):
        indices = []
        np.random.shuffle(self.classes)
        for cls in self.classes:
            cls_indices = self.class_indices[cls]
            if len(cls_indices) < self.n_positives:
                continue
            np.random.shuffle(cls_indices)
            for i in range(0, len(cls_indices), self.n_positives):
                batch = cls_indices[i:i+self.n_positives]
                if len(batch) == self.n_positives:
                    indices.extend(batch)
                    if len(indices) >= self.batch_size:
                        yield indices[:self.batch_size]
                        indices = indices[self.batch_size:]
        if len(indices) >= self.batch_size:
            yield indices[:self.batch_size]

    def __len__(self):
        return len(self.labels) // self.batch_size