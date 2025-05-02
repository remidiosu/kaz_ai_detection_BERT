import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
class DocumentBertClassifier(nn.Module):
    """
    A custom BERT-based classifier that can handle entire docs:
    1) Splits each doc into 512-token chunks using sliding window
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
        doc_embeddings = []

        for doc_text in texts:
            inputs = self.tokenizer(
                doc_text, 
                return_tensors='pt', 
                truncation=True, 
                padding='max_length', 
                max_length=self.max_length, 
                stride=self.overlap, 
                return_overflowing_tokens=True, 
                return_attention_mask=True
            ).to(device)

            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            chunk_cls_embeddings = []
            
            for i in range(input_ids.size(0)):
                output = self.bert(
                    input_ids=input_ids[i].unsqueeze(0), 
                    attention_mask=attention_mask[i].unsqueeze(0)
                )
                cls_emb = output.last_hidden_state[:, 0, :] #(1, hidden dim)
                chunk_cls_embeddings.append(cls_emb)

            # aggregate chunk embs
            chunk_cls_embeddings = torch.cat(chunk_cls_embeddings, dim=0)
            doc_emb = chunk_cls_embeddings.mean(dim=0, keepdim=True)
            doc_emb = self.dropout(doc_emb)
            
            doc_embeddings.append(doc_emb)

        # classification
        doc_embeddings = torch.cat(doc_embeddings, dim=0)
        logits = self.classifier(doc_embeddings)

        loss = None
        if labels is not None:
            labels = labels.to(device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {"loss": loss, "logits": logits}