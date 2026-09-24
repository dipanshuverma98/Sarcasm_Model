"""
predict.py — Clean, self-contained inference API.

Provides a single function `predict_sarcasm()` that accepts raw text
and returns a structured result dict.  This is the function consumed by
the Gradio demo, the unit tests, and any downstream service.

Usage:
    from sarcasm.predict import SarcasmPredictor
    predictor = SarcasmPredictor.from_checkpoint("checkpoints/best_model")
    result = predictor.predict("Oh great, another Monday!")
    print(result)
    # {'text': '...', 'label': 'SARCASTIC', 'confidence': 0.94, 'prob': 0.94}
"""

import logging
from pathlib import Path
from typing import Dict, List, Union

import torch
import yaml
from transformers import AutoTokenizer

from sarcasm.data import clean_text
from sarcasm.model import build_model

logger = logging.getLogger(__name__)


class SarcasmPredictor:
    """
    Stateful inference wrapper around a trained SarcasmClassifier.

    Encapsulates:
      - Checkpoint loading
      - Tokenisation
      - Device management
      - Result formatting

    Args:
        model:     Trained PyTorch model in eval mode.
        tokenizer: Matching HuggingFace tokenizer.
        max_length: Sequence length used during training.
        threshold:  Classification threshold (default 0.5).
        device:    Torch device.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        max_length: int = 128,
        threshold: float = 0.5,
        device: torch.device = None,
    ):
        self.model      = model
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.threshold  = threshold
        self.device     = device or torch.device("cpu")
        self.model.eval()

    # ── Class constructor ────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        device: str = "auto",
        threshold: float = 0.5,
    ) -> "SarcasmPredictor":
        """
        Load a predictor from a saved checkpoint directory.

        The directory must contain:
          - ``model.pt``       — PyTorch state dict
          - ``config.yaml``   — Training configuration
          - ``tokenizer files`` — vocab.txt, tokenizer_config.json, etc.

        Args:
            checkpoint_dir: Path to the checkpoint directory.
            device:         "auto" | "cpu" | "cuda".
            threshold:      Decision threshold.

        Returns:
            Initialised SarcasmPredictor.

        Raises:
            FileNotFoundError: If the checkpoint directory is missing.
        """
        ckpt = Path(checkpoint_dir)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: '{checkpoint_dir}'. "
                "Run train.py first to generate a checkpoint."
            )

        if device == "auto":
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            _device = torch.device(device)

        with open(ckpt / "config.yaml") as f:
            cfg = yaml.safe_load(f)

        model = build_model(cfg).to(_device)
        model.load_state_dict(
            torch.load(ckpt / "model.pt", map_location=_device)
        )
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt))

        logger.info("Loaded predictor from '%s' on %s.", checkpoint_dir, _device)
        return cls(
            model=model,
            tokenizer=tokenizer,
            max_length=cfg["model"]["max_length"],
            threshold=threshold,
            device=_device,
        )

    # ── Core inference ───────────────────────────────

    @torch.no_grad()
    def predict(self, text: str) -> Dict:
        """
        Predict whether a single text is sarcastic.

        Args:
            text: Raw input string (tweet, headline, sentence).

        Returns:
            Dictionary with keys:
              - ``text``       : cleaned input text
              - ``label``      : "SARCASTIC" or "NOT_SARCASTIC"
              - ``confidence`` : probability of the predicted class
              - ``prob``       : raw probability of sarcasm (0–1)
        """
        cleaned = clean_text(text)
        encoding = self.tokenizer(
            cleaned,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        logits = self.model(input_ids, attention_mask)
        prob   = torch.sigmoid(logits.squeeze()).item()

        is_sarcastic = prob >= self.threshold
        label        = "SARCASTIC" if is_sarcastic else "NOT_SARCASTIC"
        confidence   = prob if is_sarcastic else (1.0 - prob)

        return {
            "text":       cleaned,
            "label":      label,
            "confidence": round(confidence, 4),
            "prob":       round(prob, 4),
        }

    @torch.no_grad()
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict sarcasm for a batch of texts efficiently.

        Args:
            texts: List of raw input strings.

        Returns:
            List of prediction dictionaries (same format as ``predict``).
        """
        cleaned = [clean_text(t) for t in texts]
        encoding = self.tokenizer(
            cleaned,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        logits = self.model(input_ids, attention_mask)
        probs  = torch.sigmoid(logits.squeeze(-1)).cpu().tolist()

        results = []
        for text, prob in zip(cleaned, probs):
            is_sarcastic = prob >= self.threshold
            label        = "SARCASTIC" if is_sarcastic else "NOT_SARCASTIC"
            confidence   = prob if is_sarcastic else (1.0 - prob)
            results.append({
                "text":       text,
                "label":      label,
                "confidence": round(confidence, 4),
                "prob":       round(prob, 4),
            })
        return results
