from transformers import Trainer, TrainingArguments, set_seed, AutoTokenizer 
import torch
import numpy as np
from collections import defaultdict

from src.model import DocumentBertClassifier
from src.preprocess import prep_dataset
from src.dataset import doc_collate_fn, DocumentDataset
from src.eval_metrics import compute_metrics
from get_data.utils import get_yaml_data

class DocumentTrainer(Trainer):
    """
    Custom trainer that handles document-level evaluation properly
    """
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction_step to handle document-level predictions
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        with torch.no_grad():
            # Forward pass
            loss, logits = None, None
            with self.autocast_smart_context_manager():
                outputs = model(**inputs)
                if has_labels:
                    loss = outputs["loss"].detach()
                logits = outputs["logits"].detach()
            
            # Store unique document IDs for post-processing
            doc_indices = inputs.get("doc_indices").detach().cpu()
            
            return (loss, logits, doc_indices)
    
    def get_eval_dataloader(self, eval_dataset=None):
        """
        Return eval dataloader - unmodified from parent class
        """
        return super().get_eval_dataloader(eval_dataset)
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Override evaluation loop to handle document-level predictions
        """
        # Call parent method but store doc_indices
        eval_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )
                
        return eval_output

def train_model():
    # load YAML config file and args 
    cfg, root = get_yaml_data('train')
    batch = cfg.get('batch')
    grad_accum = cfg.get('gradient_accum')
    lr = float(cfg.get('lr'))
    epochs = cfg.get('epochs')
    seed = cfg.get('seed')
    warmup = cfg.get('warmup', 0.1)
    pretrained = cfg.get('model_name')
    output_dir = cfg.get('output_dir')
    output_dir = root / output_dir
    logs = output_dir / 'reports'
    tokenizer = AutoTokenizer.from_pretrained(pretrained)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token':'[PAD]'})
        
    # mae sure dirs exist
    output_dir.mkdir(exist_ok=True, parents=True)
    logs.mkdir(exist_ok=True, parents=True)

    set_seed(seed)

    # load data 
    train, val, _ = prep_dataset()

    # create dataset objects
    train_ds = DocumentDataset(texts=train['text'].tolist(), labels=train['label'].tolist(), tokenizer=tokenizer)
    val_ds = DocumentDataset(texts=val['text'].tolist(), labels=val['label'].tolist(), tokenizer=tokenizer)

    # instantiate the model
    model = DocumentBertClassifier(
        pretrained_name=pretrained, 
        num_labels=2
    )

    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.bert.gradient_checkpointing_enable()

    # instantiate training arguments
    args = TrainingArguments(
        output_dir=str(output_dir), 
        num_train_epochs=epochs, 
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=grad_accum,   
        eval_strategy='epoch', 
        logging_dir=str(logs), 
        logging_steps=10, 
        learning_rate=lr, 
        save_strategy='epoch', 
        seed=seed, 
        load_best_model_at_end=True, 
        metric_for_best_model='precision',
        warmup_ratio=warmup, 
        weight_decay=0.01,
        dataloader_num_workers=4,
        fp16=True, 
        dataloader_drop_last=True,
    )

    # Get document IDs from validation dataset
    val_doc_ids = list(range(len(val['label'])))
    val_labels = val['label'].tolist()
    
    # Modified compute_metrics lambda
    metrics_fn = lambda eval_pred: compute_metrics(eval_pred, val_labels)

    # Use custom trainer
    trainer = DocumentTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=doc_collate_fn,
        compute_metrics=metrics_fn,
    )

    trainer.train()
    trainer.save_model(output_dir)

    return trainer