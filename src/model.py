import torch
import torch.nn as nn
from transformers import AutoModel

class DocumentBertClassifier(nn.Module):
    def __init__(self, pretrained_name, num_labels=2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, doc_indices, labels=None):
        # Forward pass through BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_embeds = outputs.last_hidden_state[:, 0, :]  # (total_chunks, hidden_size)
        
        # Group embeddings by document
        unique_docs = torch.unique(doc_indices)
        doc_embeds = []
        document_labels = []
        
        # Process each document separately
        for doc_idx in unique_docs:
            # Find all chunks belonging to this document
            doc_mask = (doc_indices == doc_idx)
            # Get embeddings for this document's chunks
            doc_chunk_embeds = cls_embeds[doc_mask]
            # Average the embeddings for this document
            doc_embed = torch.mean(doc_chunk_embeds, dim=0)
            doc_embeds.append(doc_embed)
            
            # Get the corresponding label for this document
            if labels is not None:
                chunk_indices = torch.where(doc_mask)[0]
                document_labels.append(labels[chunk_indices[0]].item())  
        
        # Stack all document embeddings
        doc_embeds = torch.stack(doc_embeds)
        
        # Classification
        logits = self.classifier(doc_embeds)
        
        loss = None
        if labels is not None:
            # Convert document labels to tensor
            document_labels = torch.tensor(document_labels, device=logits.device, dtype=torch.long)
            # Calculate loss
            loss = nn.CrossEntropyLoss()(logits, document_labels)
        
        return {'loss': loss, 'logits': logits}