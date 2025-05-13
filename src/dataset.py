from torch.utils.data import Dataset
import torch

class DocumentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512, overlap=50):
        self.texts = texts  
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.overlap = overlap
        self.all_chunks = []
        self.doc_indices = []
        self.all_masks = []
        
        for doc_idx, (text, label) in enumerate(zip(texts, labels)):
            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            chunks, masks = self._split_into_chunks(tokens)
            self.all_chunks.extend(chunks)
            self.all_masks.extend(masks)
            self.doc_indices.extend([doc_idx] * len(chunks))  

    def __len__(self):
        return len(self.all_chunks)  
    
    def _preprocess(self, texts, labels):
        data = []
        for text, label in zip(texts, labels):
            tokens = self.tokenizer(text, add_special_tokens=False)['input_ids']
            chunks = self._split_into_chunks(tokens)
            data.append({'chunks':chunks, 'label':label})
        return data
    
    def _split_into_chunks(self, tokens):
        chunks = []
        attention_masks = []
        start = 0 
        body_max = self.max_length - 2  # Reserve space for [CLS] and [SEP]
        
        while start < len(tokens):
            end = start + body_max
            chunk = tokens[start:end]
            
            # Add special tokens
            chunk = self.tokenizer.build_inputs_with_special_tokens(chunk)
            
            # Create attention mask before padding
            real_tokens = len(chunk)
            attention_mask = [1] * real_tokens
            
            # Pad to max_length
            padding = [self.tokenizer.pad_token_id] * (self.max_length - real_tokens)
            chunk += padding
            
            # Extend attention mask for padding
            attention_mask += [0] * (self.max_length - real_tokens)
            
            chunks.append(chunk)
            attention_masks.append(attention_mask)
            start = end - self.overlap if (end - self.overlap) > start else end
        
        return chunks, attention_masks

    def __getitem__(self, index):
        return {
            "input_ids": self.all_chunks[index],
            "attention_mask": self.all_masks[index],  # New
            "labels": self.labels[self.doc_indices[index]],
            "doc_indices": self.doc_indices[index]
        }

def doc_collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]  # New
    labels = [item["labels"] for item in batch]
    doc_indices = [item["doc_indices"] for item in batch]
    
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),  # New
        "labels": torch.tensor(labels, dtype=torch.long),
        "doc_indices": torch.tensor(doc_indices, dtype=torch.long)
    }