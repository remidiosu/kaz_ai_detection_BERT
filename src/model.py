import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
class DocumentBertClassifier(nn.Module):
    """
    A custom BERT-based classifier that can handle entire docs:
    1) Splits each doc into 512-token chunks
    2) Passes them through BERT
    3) Aggregates chunk embeddings into a single doc embedding
    4) Outputs a single classification for the doc
    """
    def __init__(self, pretrained_name, num_labels=2,
                 max_length=512, overlap=50):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.bert = AutoModel.from_pretrained(pretrained_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        self.max_length = max_length
        self.overlap = overlap  
        self.dropout = nn.Dropout(0.1)

    def forward(self, texts, labels=None):
        device = next(self.parameters()).device
        # We'll store one doc embedding per item in the batch
        doc_embeddings = []

        for doc_text in texts:
            # 1) Tokenize the entire doc
            tokens = self.tokenizer.tokenize(doc_text)
            chunk_cls_embeddings = []
            start = 0

            # 2) Slide over doc with chunk size = max_length and overlap
            while start < len(tokens):
                end = start + self.max_length
                chunk_tokens = tokens[start:end]
                # Convert tokens to string
                chunk_str = self.tokenizer.convert_tokens_to_string(chunk_tokens)

                # Convert chunk_str to model inputs
                inputs = self.tokenizer(
                    chunk_str,
                    return_tensors="pt",
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length
                ).to(device)

                # 3) Pass chunk through BERT
                outputs = self.bert(**inputs)
                # Take [CLS] embedding (index 0)
                cls_emb = outputs.last_hidden_state[:, 0, :]  # shape: (1, hidden_dim)
                chunk_cls_embeddings.append(cls_emb)

                if end >= len(tokens):
                    break
                start = end - self.overlap

            # 4) Aggregate chunk embeddings
            # shape of chunk_cls_embeddings: list of (1, hidden_dim)
            chunk_cls_embeddings = torch.cat(chunk_cls_embeddings, dim=0)  # (num_chunks, hidden_dim)
            doc_emb = chunk_cls_embeddings.mean(dim=0, keepdim=True)       # (1, hidden_dim)

            doc_embeddings.append(doc_emb)

        # (batch_size, hidden_dim)
        doc_embeddings = torch.cat(doc_embeddings, dim=0)

        # 5) Classifier
        logits = self.classifier(doc_embeddings)
        loss = None
        if labels is not None:
            labels = labels.to(device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}