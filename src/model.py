import torch
import torch.nn as nn
from transformers import AutoModel

class DocumentBertClassifier(nn.Module):
    def __init__(self, pretrained_name, num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, doc_indices, labels=None):
        # Forward all chunks
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeds = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # Group embeddings by document
        unique_docs, inverse_indices = torch.unique(doc_indices, return_inverse=True)
        doc_embeds = torch.zeros(
            (unique_docs.size(0), cls_embeds.size(1)), 
            device=cls_embeds.device
        )
        doc_embeds.scatter_add_(
            0, 
            inverse_indices.unsqueeze(1).expand(-1, cls_embeds.size(1)), 
            cls_embeds
        )
        doc_counts = torch.bincount(inverse_indices)
        doc_embeds /= doc_counts.unsqueeze(1)  # Average pooling
        
        logits = self.classifier(doc_embeds)
        loss = None
        if labels is not None: 
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {'loss':loss, 'logits':logits}