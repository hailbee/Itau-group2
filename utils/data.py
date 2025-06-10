import torch
from torch.utils.data import Dataset

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
