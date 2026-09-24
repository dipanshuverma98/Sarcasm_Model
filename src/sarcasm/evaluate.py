"""
evaluate.py — Deep model evaluation with error analysis.

Produces:
  - Classification report (precision / recall / F1 per class)
  - Confusion matrix plot
  - Calibration curve
  - Error analysis: top misclassified examples with confidence scores
  - Length-stratified accuracy (short vs. long tweets)
  - Keyword frequency analysis in FP / FN buckets

Usage:
    python -m sarcasm.evaluate --checkpoint checkpoints/best_model
"""

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from transformers import AutoTokenizer

from sarcasm.data import load_dataframe, make_splits, build_dataloaders
from sarcasm.model import build_model

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

LABEL_NAMES = ["Not Sarcastic", "Sarcastic"]


# ──────────────────────────────────────────────
#  Plot helpers
# ──────────────────────────────────────────────

def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    save_path: Path,
) -> None:
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"],
        ["d", ".2%"],
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved → %s", save_path)


def plot_calibration_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: Path,
) -> None:
    frac_pos, mean_pred = calibration_curve(labels, probs, n_bins=10)

    plt.figure(figsize=(7, 6))
    plt.plot(mean_pred, frac_pos, "o-", label="BERT Sarcasm Classifier")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Calibration curve saved → %s", save_path)


def plot_roc_curve(
    labels: np.ndarray,
    probs: np.ndarray,
    save_path: Path,
) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Sarcasm Detection")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("ROC curve saved → %s", save_path)


# ──────────────────────────────────────────────
#  Error analysis
# ──────────────────────────────────────────────

def error_analysis(
    texts: List[str],
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    save_path: Path,
    top_n: int = 20,
) -> None:
    """
    Produce a detailed error report:
      - Top N false positives (most confidently wrong: predicted sarcastic, actually not)
      - Top N false negatives (predicted not-sarcastic, actually sarcastic)
      - Most common trigrams in each error bucket

    Args:
        texts:     Raw text samples.
        labels:    Ground-truth labels.
        preds:     Model predictions.
        probs:     Model confidence scores.
        save_path: Path to save the markdown report.
        top_n:     Number of examples to include per error type.
    """
    texts = np.array(texts)

    fp_mask = (preds == 1) & (labels == 0)   # False Positives
    fn_mask = (preds == 0) & (labels == 1)   # False Negatives

    def _top_errors(mask, confidence_fn):
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return []
        sorted_idxs = idxs[np.argsort(confidence_fn(idxs))[::-1]]
        return sorted_idxs[:top_n]

    fp_idxs = _top_errors(fp_mask, lambda i: probs[i])
    fn_idxs = _top_errors(fn_mask, lambda i: 1 - probs[i])

    def _top_trigrams(idxs: np.ndarray, n: int = 10) -> List[str]:
        from collections import Counter
        words = " ".join(texts[idxs]).lower().split()
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        return [tg for tg, _ in Counter(trigrams).most_common(n)]

    lines = [
        "# Error Analysis Report\n",
        f"Total errors: **{fp_mask.sum() + fn_mask.sum()}** "
        f"({fp_mask.sum()} FP, {fn_mask.sum()} FN)\n",
        f"Error rate: **{(fp_mask.sum() + fn_mask.sum()) / len(labels):.2%}**\n\n",

        "## False Positives (predicted Sarcastic, actually Not)\n",
        "Common trigrams: " + ", ".join(f"`{tg}`" for tg in _top_trigrams(fp_idxs)) + "\n\n",
        "| Confidence | Text |\n|---|---|\n",
    ]
    for idx in fp_idxs:
        text_snippet = texts[idx][:120].replace("|", "∣")
        lines.append(f"| {probs[idx]:.3f} | {text_snippet} |\n")

    lines += [
        "\n## False Negatives (predicted Not Sarcastic, actually Sarcastic)\n",
        "Common trigrams: " + ", ".join(f"`{tg}`" for tg in _top_trigrams(fn_idxs)) + "\n\n",
        "| Confidence | Text |\n|---|---|\n",
    ]
    for idx in fn_idxs:
        text_snippet = texts[idx][:120].replace("|", "∣")
        lines.append(f"| {1-probs[idx]:.3f} | {text_snippet} |\n")

    # Length-stratified accuracy
    lengths = np.array([len(t.split()) for t in texts])
    lines.append("\n## Accuracy by Tweet Length\n\n")
    lines.append("| Length bucket | # samples | Accuracy |\n|---|---|---|\n")
    buckets = [(1, 10), (11, 20), (21, 40), (41, 80), (81, 300)]
    for lo, hi in buckets:
        mask = (lengths >= lo) & (lengths <= hi)
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == labels[mask]).mean()
        lines.append(f"| {lo}–{hi} words | {mask.sum()} | {acc:.3%} |\n")

    with open(save_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    logger.info("Error analysis saved → %s", save_path)


# ──────────────────────────────────────────────
#  Main evaluation entry point
# ──────────────────────────────────────────────

def evaluate_checkpoint(checkpoint_dir: str, output_dir: str) -> None:
    """
    Load a saved checkpoint and run full evaluation on the held-out test set.

    Args:
        checkpoint_dir: Path containing model.pt, tokenizer files, config.yaml.
        output_dir:     Directory to write evaluation artefacts.
    """
    ckpt = Path(checkpoint_dir)
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config and rebuild model
    with open(ckpt / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(ckpt / "model.pt", map_location=device))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))

    # Reload test split
    df = load_dataframe(cfg["data"]["csv_path"])
    _, _, test_df = make_splits(
        df,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        seed=cfg["training"]["seed"],
    )
    _, _, test_loader = build_dataloaders(
        test_df, test_df, test_df,
        tokenizer=tokenizer,
        batch_size=cfg["training"]["batch_size"],
        max_length=cfg["model"]["max_length"],
    )

    # Collect predictions
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            probs  = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            preds  = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch["label"].numpy())

    probs  = np.array(all_probs)
    preds  = np.array(all_preds)
    labels = np.array(all_labels)

    # ── Print classification report ──────────
    logger.info("\n%s", classification_report(labels, preds, target_names=LABEL_NAMES))

    # ── Generate plots ───────────────────────
    plot_confusion_matrix(labels, preds, out / "confusion_matrix.png")
    plot_calibration_curve(labels, probs, out / "calibration_curve.png")
    plot_roc_curve(labels, probs, out / "roc_curve.png")

    # ── Error analysis ───────────────────────
    error_analysis(
        texts=test_df["text"].tolist(),
        labels=labels,
        preds=preds,
        probs=probs,
        save_path=out / "error_analysis.md",
    )

    # ── Summary metrics ──────────────────────
    from sklearn.metrics import accuracy_score, f1_score
    summary = {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds),
        "roc_auc":  roc_auc_score(labels, probs),
    }
    logger.info("Summary: %s", summary)

    with open(out / "metrics.yaml", "w") as f:
        yaml.dump({k: float(v) for k, v in summary.items()}, f)

    logger.info("All evaluation artefacts written to %s", out)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved sarcasm detection checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory.")
    parser.add_argument("--output-dir", default="eval_output", help="Where to save plots & reports.")
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
