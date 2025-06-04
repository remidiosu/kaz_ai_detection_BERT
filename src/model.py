import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class DocumentBertClassifier(nn.Module):
    """
    BERT‐based classifier for long documents, avoiding redundant tokenization.
    Now forward() expects:
      - input_ids:   a list of 1D tensors (each = full doc’s token IDs, no special tokens),
      - attention_mask: a list of 1D tensors (each = full doc’s attention mask),
      - labels:      (optional) tensor of shape (batch_size,).
    It then does chunking with [CLS]/[SEP], padding, BERT, and averages chunk‐CLS embeddings.
    """
    def __init__(self, pretrained_name: str, num_labels: int = 2, max_length: int = 512, overlap: int = 50):
        super().__init__()
        # We still need a tokenizer to build special tokens and pad each chunk
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.bert = AutoModel.from_pretrained(pretrained_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        self.max_length = max_length
        self.overlap = overlap

    def forward(self,
                input_ids: list[torch.Tensor],
                attention_mask: list[torch.Tensor],
                labels: torch.Tensor = None):
        """
        Args:
          input_ids:   List[Tensor] of shape (n_tokens,) for each document (no special tokens).
          attention_mask: List[Tensor] of shape (n_tokens,) for each document.
          labels:      (optional) Tensor of shape (batch_size,).

        Returns:
          {"loss": loss_or_None, "logits": (batch_size, num_labels)}
        """
        device = next(self.parameters()).device
        doc_embeddings = []

        for full_ids, full_mask in zip(input_ids, attention_mask):
            # Move to device
            full_ids = full_ids.to(device)
            full_mask = full_mask.to(device)

            chunk_embs = []
            start = 0
            body_max = self.max_length - 2  # reserve for [CLS] & [SEP]

            # Slide window over the document tokens
            while start < full_ids.size(0):
                end = start + body_max
                # 1) slice out this chunk of token IDs
                chunk_ids = full_ids[start:end]
                chunk_mask = full_mask[start:end]

                # 2) build special‐token sequence around chunk_ids
                chunk_with_special = self.tokenizer.build_inputs_with_special_tokens(chunk_ids.tolist())

                # 3) pad to max_length
                padded = self.tokenizer.pad(
                    {"input_ids": [chunk_with_special]},
                    padding="max_length",
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(device)
                # padded["input_ids"].shape == (1, max_length)

                # 4) forward through BERT
                outputs = self.bert(**padded)
                # take the [CLS] embedding (shape (1, hidden_size))
                cls_emb = outputs.last_hidden_state[:, 0, :]  # (1, hidden_dim)
                chunk_embs.append(cls_emb)

                if end >= full_ids.size(0):
                    break
                # overlap: shift start to end - overlap
                start = end - self.overlap

            # 5) stack all chunk CLS embeddings and average them
            chunk_embs = torch.cat(chunk_embs, dim=0)         # (n_chunks, hidden_dim)
            doc_emb = chunk_embs.mean(dim=0, keepdim=True)    # (1, hidden_dim)
            doc_emb = self.dropout(doc_emb)                   # (1, hidden_dim)
            doc_embeddings.append(doc_emb)

        # stack to shape (batch_size, hidden_dim)
        doc_embeddings = torch.cat(doc_embeddings, dim=0)
        logits = self.classifier(doc_embeddings)              # (batch_size, num_labels)

        loss = None
        if labels is not None:
            labels = labels.to(device)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}
