from torch.utils.data import Dataset
from transformers import AutoTokenizer
from get_data.utils import get_yaml_data
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
    cfg, root = get_yaml_data('train')
    pretrained = cfg.get('model_name')
    texts, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    input_ids_list = []
    attention_mask_list = []

    for doc in texts:
        enc = tokenizer(
            doc,
            add_special_tokens=False, 
            return_attention_mask=True,
            return_tensors="pt"
        )
        # enc["input_ids"].shape == (1, n_tokens)
        # we want the 1D tensor (n_tokens,)
        input_ids_list.append(enc["input_ids"].squeeze(0))
        attention_mask_list.append(enc["attention_mask"].squeeze(0))

    return {
        "input_ids": input_ids_list,           # list of 1D LongTensors
        "attention_mask": attention_mask_list, # list of 1D LongTensors
        "labels": labels                       # (batch_size,)
    }
