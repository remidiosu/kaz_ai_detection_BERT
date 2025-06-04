from transformers import Trainer, TrainingArguments, set_seed, AutoTokenizer
import torch

from src.model import DocumentBertClassifier
from src.preprocess import prep_dataset
from src.dataset import DocumentDataset, doc_collate_fn
from src.eval_metrics import compute_metrics
from get_data.utils import get_yaml_data

def train_model():
    # 1. load YAML config file and args 
    cfg, root = get_yaml_data('train')
    batch = cfg.get('batch')
    grad_accum = cfg.get('gradient_accum')
    lr = float(cfg.get('lr'))
    epochs = cfg.get('epochs')
    seed = cfg.get('seed')
    pretrained = cfg.get('model_name')
    output_dir = cfg.get('output_dir')
    output_dir = root / output_dir
    logs = output_dir / 'reports'

    output_dir.mkdir(exist_ok=True, parents=True)
    logs.mkdir(exist_ok=True, parents=True)
    set_seed(seed)

    # 2. load data splits
    train_df, val_df, _ = prep_dataset()

    # 3. create dataset objects 
    train_ds = DocumentDataset(texts=train_df['text'].tolist(), labels=train_df['label'].tolist())
    val_ds   = DocumentDataset(texts=val_df['text'].tolist(),   labels=val_df['label'].tolist())

    # 4. instantiate the model
    model = DocumentBertClassifier(
        pretrained_name=pretrained, 
        num_labels=2, 
        max_length=512, 
        overlap=50
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    model.bert.gradient_checkpointing_enable()

    # 5. training arguments
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
        metric_for_best_model='f1',
        warmup_ratio=0.1,
        weight_decay=0.01,
        dataloader_num_workers=4,
        fp16=True
    )

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
