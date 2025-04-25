from torch.utils.data import Dataset
import torch

class DocumentDataset(Dataset):
    """Each sample is one document"""
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        return self.texts[index], self.labels[index]
    
def doc_collate_fn(batch):
    texts, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    return {
        "texts": list(texts), 
        "labels": labels
    }