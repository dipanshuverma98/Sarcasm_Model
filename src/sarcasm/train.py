"""
train.py — Full training loop with validation, early stopping, and checkpointing.

Usage (from project root):
    python -m sarcasm.train --config configs/bert_config.yaml

Features:
  - Mixed-precision training (fp16) for faster GPU utilisation
  - Linear warmup + cosine decay learning-rate schedule
  - Early stopping on validation F1
  - Best-model checkpointing (saves to checkpoints/)
  - Per-epoch metrics logged to stdout and optionally to Weights & Biases
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, accuracy_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from sarcasm.data import load_dataframe, make_splits, build_dataloaders
from sarcasm.model import build_model

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Config loader
# ──────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Load YAML config and expand any environment-variable references."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # Allow ${ENV_VAR} in string values
    def _expand(obj):
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        return obj
    return _expand(cfg)


# ──────────────────────────────────────────────
#  One epoch helpers
# ──────────────────────────────────────────────

def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Run one full training epoch. Returns {'loss': float, 'acc': float}."""
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for step, batch in enumerate(loader, 1):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()

        with autocast():
            loss, logits = model(input_ids, attention_mask, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).long().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if step % 100 == 0:
            running_loss = total_loss / step
            logger.info("  step %4d | loss %.4f", step, running_loss)

    return {
        "loss": total_loss / len(loader),
        "acc":  accuracy_score(all_labels, all_preds),
        "f1":   f1_score(all_labels, all_preds, average="binary"),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate on a dataloader. Returns loss, acc, f1."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        with autocast():
            loss, logits = model(input_ids, attention_mask, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    return {
        "loss": total_loss / len(loader),
        "acc":  accuracy_score(all_labels, all_preds),
        "f1":   f1_score(all_labels, all_preds, average="binary"),
        "probs":  np.array(all_probs),
        "preds":  np.array(all_preds),
        "labels": np.array(all_labels),
    }


# ──────────────────────────────────────────────
#  Main training routine
# ──────────────────────────────────────────────

def train(cfg: dict) -> None:
    """
    End-to-end training pipeline driven by a config dictionary.

    1. Set random seeds for reproducibility
    2. Load and split data
    3. Build tokenizer, dataloaders, model
    4. Run training loop with early stopping
    5. Save best model + tokenizer
    """
    # ── Reproducibility ──────────────────────
    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Device ───────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ── Data ─────────────────────────────────
    df = load_dataframe(cfg["data"]["csv_path"])
    train_df, val_df, test_df = make_splits(
        df,
        train_ratio=cfg["data"]["train_ratio"],
        val_ratio=cfg["data"]["val_ratio"],
        seed=seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df,
        tokenizer=tokenizer,
        batch_size=cfg["training"]["batch_size"],
        max_length=cfg["model"]["max_length"],
    )

    # ── Model ────────────────────────────────
    model = build_model(cfg).to(device)
    logger.info("Model loaded with %s trainable parameters.", f"{model.get_trainable_params():,}")

    # ── Optimiser & Schedule ─────────────────
    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg["training"]["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(params, lr=cfg["training"]["learning_rate"])

    total_steps   = len(train_loader) * cfg["training"]["epochs"]
    warmup_steps  = int(total_steps * cfg["training"].get("warmup_ratio", 0.06))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler    = GradScaler()

    # ── Checkpoint directory ─────────────────
    ckpt_dir = Path(cfg["training"].get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────
    best_val_f1   = 0.0
    patience      = cfg["training"].get("patience", 3)
    patience_left = patience

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        logger.info("=" * 60)
        logger.info("Epoch %d / %d", epoch, cfg["training"]["epochs"])
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device
        )
        val_metrics = evaluate(model, val_loader, device)

        elapsed = time.time() - t0
        logger.info(
            "Train  loss=%.4f  acc=%.4f  f1=%.4f",
            train_metrics["loss"], train_metrics["acc"], train_metrics["f1"],
        )
        logger.info(
            "Val    loss=%.4f  acc=%.4f  f1=%.4f  (%.0fs)",
            val_metrics["loss"], val_metrics["acc"], val_metrics["f1"], elapsed,
        )

        # ── Save best model ──────────────────
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_left = patience

            save_path = ckpt_dir / "best_model"
            save_path.mkdir(exist_ok=True)

            # Save model weights
            torch.save(model.state_dict(), save_path / "model.pt")
            # Save tokenizer (needed for inference)
            tokenizer.save_pretrained(save_path)
            # Save config snapshot for reproducibility
            with open(save_path / "config.yaml", "w") as f:
                yaml.dump(cfg, f)

            logger.info("✓ Best model saved  (val F1=%.4f)", best_val_f1)
        else:
            patience_left -= 1
            logger.info(
                "No improvement. Patience: %d / %d", patience - patience_left, patience
            )
            if patience_left == 0:
                logger.info("Early stopping triggered.")
                break

    # ── Final test evaluation ─────────────────
    logger.info("=" * 60)
    logger.info("Loading best checkpoint for final test evaluation …")

    model.load_state_dict(torch.load(ckpt_dir / "best_model" / "model.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, device)

    logger.info(
        "TEST   loss=%.4f  acc=%.4f  f1=%.4f",
        test_metrics["loss"], test_metrics["acc"], test_metrics["f1"],
    )

    # Save test predictions for error analysis
    preds_path = ckpt_dir / "test_predictions.npz"
    np.savez(
        preds_path,
        texts=test_df["text"].values,
        labels=test_metrics["labels"],
        preds=test_metrics["preds"],
        probs=test_metrics["probs"],
    )
    logger.info("Test predictions saved to %s", preds_path)


# ──────────────────────────────────────────────
#  CLI entry point
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train sarcasm detection model.")
    parser.add_argument(
        "--config",
        default="configs/bert_config.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info("Config loaded from: %s", args.config)
    train(cfg)


if __name__ == "__main__":
    main()
