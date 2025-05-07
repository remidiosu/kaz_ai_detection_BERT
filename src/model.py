import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class DocumentBertClassifier(nn.Module):
    """
    BERT-based classifier for long documents, avoiding redundant tokenization.
    Steps:
    1) Tokenize full document once (no special tokens)
    2) Slide window over token IDs with overlap, adding special tokens per chunk
    3) Pad each chunk to max_length and pass through BERT
    4) Aggregate [CLS] embeddings and classify
    
    """
    def __init__(self, pretrained_name, num_labels=2, max_length=512, overlap=50):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.bert = AutoModel.from_pretrained(pretrained_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.max_length = max_length
        self.overlap = overlap

    def forward(self, texts, labels=None):
        device = next(self.parameters()).device
        doc_embeddings = []

        for doc_text in texts:
            # 1) Tokenize full doc without special tokens
            encoding = self.tokenizer(
                doc_text,
                add_special_tokens=False,
                return_attention_mask=True,
                return_tensors="pt"
            )
            full_ids = encoding["input_ids"][0]
            full_mask = encoding["attention_mask"][0]

            chunk_embs = []
            start = 0
            # Reserve 2 tokens for [CLS] and [SEP]
            body_max = self.max_length - 2
            while start < full_ids.size(0):
                end = start + body_max
                # slice tokens
                chunk_ids = full_ids[start:end]

                # 2) add special tokens
                chunk_with_special = self.tokenizer.build_inputs_with_special_tokens(
                    chunk_ids.tolist()
                )

                # 3) pad to max_length
                inputs = self.tokenizer.pad(
                    {"input_ids": [chunk_with_special]},
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(device)

                # 4) forward through BERT
                outputs = self.bert(**inputs)
                cls_emb = outputs.last_hidden_state[:, 0, :]  # (1, hidden_dim)
                chunk_embs.append(cls_emb)

                if end >= full_ids.size(0):
                    break
                start = end - self.overlap

            # 5) aggregate chunk embeddings
            chunk_embs = torch.cat(chunk_embs, dim=0)        # (n_chunks, hidden_dim)
            doc_emb = chunk_embs.mean(dim=0, keepdim=True)  # (1, hidden_dim)
            doc_emb = self.dropout(doc_emb)
            doc_embeddings.append(doc_emb)

        # (batch_size, hidden_dim)
        doc_embeddings = torch.cat(doc_embeddings, dim=0)
        logits = self.classifier(doc_embeddings)

        loss = None
        if labels is not None:
            labels = labels.to(device)
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}
