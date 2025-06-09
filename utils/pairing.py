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