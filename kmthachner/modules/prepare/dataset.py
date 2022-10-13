from torch.utils.data import Dataset

class ConllDataset(Dataset):
    def __init__(self, dataset, preprocessing):
        self.dataset = dataset
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        tokens = self.dataset[index]["tokens"]
        tags = self.dataset[index]["ner_tags"]
        converted_sample = self.preprocessing.preprocess(tokens, tags)
        return converted_sample

class PredictDataset(Dataset):
    # Dataset for predict sequences.
    def __init__(self, dataset, preprocessing):
        self.dataset = dataset
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        tokens = self.dataset[index]
        converted_sample = self.preprocessing.preprocess(tokens, None)
        return converted_sample
