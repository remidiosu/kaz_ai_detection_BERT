from transformers import Trainer, TrainingArguments, set_seed, AutoTokenizer 
import torch

from src.model import DocumentBertClassifier
from src.preprocess import prep_dataset
from src.dataset import doc_collate_fn, DocumentDataset
from src.eval_metrics import compute_metrics
from get_data.utils import get_yaml_data

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
        fp16=True
    )

    # Trainer
    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset=train_ds, 
        eval_dataset=val_ds, 
        data_collator=doc_collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)

    return trainer

