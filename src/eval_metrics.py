from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np 
import torch

def compute_metrics(eval_pred, val_labels):
    logits = eval_pred.predictions
    preds = np.argmax(logits, axis=1)
    
    # Get unique document IDs from the validation dataset
    if len(preds) != len(val_labels):
        # Use the number of predictions we have
        actual_labels = val_labels[:len(preds)] if len(preds) < len(val_labels) else val_labels
    else:
        actual_labels = val_labels
    
    return {
        "accuracy": accuracy_score(actual_labels[:len(preds)], preds),
        "f1": f1_score(actual_labels[:len(preds)], preds, average="binary"),
        "precision": precision_score(actual_labels[:len(preds)], preds, average="binary"),
        "recall": recall_score(actual_labels[:len(preds)], preds, average="binary")
    }