import torch
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self, anchor_data, positive_data, negative_data, tokenizer):
        self.anchor_data = anchor_data
        self.positive_data = positive_data
        self.negative_data = negative_data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.anchor_data)

    def __getitem__(self, idx):
        anchor = self.tokenizer(self.anchor_data[idx], return_tensors='pt', padding='max_length', truncation=True)
        positive = self.tokenizer(self.positive_data[idx], return_tensors='pt', padding='max_length', truncation=True)
        negative = self.tokenizer(self.negative_data[idx], return_tensors='pt', padding='max_length', truncation=True)

        # remove batch dimension
        return { 
            'anchor': {k: v.squeeze(0) for k, v in anchor.items()},
            'positive': {k: v.squeeze(0) for k, v in positive.items()},
            'negative': {k: v.squeeze(0) for k, v in negative.items()}
        }
