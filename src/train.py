from transformers import Trainer, TrainingArguments

from src.preprocess import prep_dataset
from src.dataset import doc_collate_fn, DocumentDataset
def train_model():
    # load data
    train, val, test = prep_dataset()

    # create dataset objects
    train_ds = DocumentDataset(texts=train['text'].tolist(), labels=train['label'])
    val_ds = DocumentDataset(texts=val['text'].tolist(), labels=val['label'])
    test_ds = DocumentDataset(texts=test['text'].tolist(), labels=test['label'])