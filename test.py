from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from safetensors.torch import load_file
from transformers import Trainer, TrainingArguments
from src.model import DocumentBertClassifier
from src.dataset import DocumentDataset, doc_collate_fn
from src.preprocess import prep_dataset
from src.eval_metrics import compute_metrics
from get_data.utils import get_yaml_data
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)


def plot_and_save_roc(y_true, y_scores, out_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0,1],[0,1],'--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(out_path / 'roc_curve.png', dpi=200)
    plt.close()


def plot_and_save_pr(y_true, y_scores, out_path: Path):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision, label=f'AP = {ap:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision–Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(out_path / 'pr_curve.png', dpi=200)
    plt.close()


def plot_and_save_confusion(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.savefig(out_path / 'confusion_matrix.png', dpi=200)
    plt.close()


def plot_loss_curves(trainer, out_path: Path):
    # requires that trainer.state.log_history contains 'loss' and 'eval_loss'
    import pandas as pd
    logs = pd.DataFrame(trainer.state.log_history)
    if {'loss','eval_loss'}.issubset(logs.columns):
        plt.figure()
        plt.plot(logs['loss'], label='train loss')
        plt.plot(logs['eval_loss'], label='eval loss')
        plt.xlabel('Logging step')
        plt.ylabel('Loss')
        plt.title('Training vs. Evaluation Loss')
        plt.legend()
        plt.savefig(out_path / 'loss_curves.png', dpi=200)
        plt.close()


def main():
    # 0) Config
    cfg, ROOT = get_yaml_data('train')
    batch = cfg.get('batch', 4)
    pretrained_name = cfg.get('model_name')
    output_dir = Path(cfg.get('output_dir'))
    checkpoint = output_dir / "checkpoint-680"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Model
    model = DocumentBertClassifier(
        pretrained_name=pretrained_name,
        num_labels=2,
        max_length=512,
        overlap=50
    ).to(device)
    sd = load_file(str(checkpoint/"model.safetensors"), device=device)
    model.load_state_dict(sd, strict=False)

    # 2) Trainer (for eval + access to log_history)
    args = TrainingArguments(
        output_dir=str(checkpoint),
        per_device_eval_batch_size=batch,
        do_train=False,
        do_eval=False,
        logging_dir=str(checkpoint/"reports"),
    )
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=doc_collate_fn,
        compute_metrics=compute_metrics
    )

    # 3) Data
    _, _, test = prep_dataset()
    test_ds = DocumentDataset(test["text"].tolist(), test["label"].tolist())

    # 4) Predict
    print("Running test evaluation…")
    out = trainer.predict(test_ds)
    metrics = out.metrics
    print("Test metrics:", metrics)

    # 5) Save raw metrics
    report_dir = ROOT/'reports'/output_dir.name
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir/'test_metrics.txt').write_text(str(metrics))

    # 6) Prepare y_true, y_pred, y_score
    y_true = out.label_ids
    if out.predictions.ndim == 2:
        # assume logits for two classes
        y_scores = torch.softmax(torch.tensor(out.predictions), dim=1)[:,1].numpy()
        y_pred = np.argmax(out.predictions, axis=1)
    else:
        # single logit
        y_scores = out.predictions.ravel()
        y_pred = (y_scores > 0.5).astype(int)

    # 7) Plot & save figures
    plot_and_save_roc(y_true, y_scores, report_dir)
    plot_and_save_pr(y_true, y_scores, report_dir)
    plot_and_save_confusion(y_true, y_pred, report_dir)
    plot_loss_curves(trainer, report_dir)

    print(f"All reports saved to {report_dir}")

if __name__ == "__main__":
    main()
