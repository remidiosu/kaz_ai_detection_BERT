from transformers import Trainer, TrainingArguments, set_seed

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
    lr = cfg.get('lr')
    epochs = cfg.get('epochs')
    seed = cfg.get('seed')
    pretrained = cfg.get('model_name')
    output_dir = cfg.get('output_dir')
    output_dir = root / output_dir
    logs = output_dir / 'reports'

    # mae sure dirs exist
    output_dir.mkdir(exist_ok=True, parents=True)
    logs.mkdir(exist_ok=True, parents=True)

    set_seed(seed)

    # load data 
    train, val, test = prep_dataset()

    # create dataset objects
    train_ds = DocumentDataset(texts=train['text'].tolist(), labels=train['label'].tolist())
    val_ds = DocumentDataset(texts=val['text'].tolist(), labels=val['label'].tolist())
    test_ds = DocumentDataset(texts=test['text'].tolist(), labels=test['label'].tolist())

    # instantiate the model
    model = DocumentBertClassifier(
        pretrained_name=pretrained, 
        num_labels=2, 
        max_length=512, 
        overlap=50
    )

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
        metric_for_best_model='accuracy',
        warmup_ratio=0.1, 
        weight_decay=0.01
    )

    # Trainer
    trainer = Trainer(
        model=model, 
        args=args, 
        train_dataset=train_ds, 
        eval_dataset=val_ds, 
        data_collator=doc_collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=model.tokenizer
    )

    trainer.train()
    trainer.save_model(output_dir)

    test_rep = logs / f'test_{pretrained}.txt'
    with open(test_rep, 'a') as log:
        test_metrics = trainer.predict(test_ds).metrics
        print("Test set metrics:", test_metrics)
        log.write(f"Test set metrics: {test_metrics}")

    return trainer, test_metrics

