"""
test_predict.py — Unit tests for the SarcasmPredictor inference API.

These tests use a tiny random-weight model (no checkpoint required)
so the CI pipeline never needs a GPU or a real trained model.

Run with:
    pytest tests/ -v
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

from sarcasm.data import clean_text
from sarcasm.model import BERTSarcasmClassifier, BiLSTMClassifier


# ──────────────────────────────────────────────
#  clean_text unit tests
# ──────────────────────────────────────────────

class TestCleanText:
    """Tests for the text normalisation utility."""

    def test_lowercases_text(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_removes_urls(self):
        result = clean_text("Check this out https://example.com right?")
        assert "https" not in result
        assert "example.com" not in result

    def test_anonymises_mentions(self):
        result = clean_text("Hey @JohnDoe, what do you think?")
        assert "@johndoe" not in result
        assert "@user" in result

    def test_collapses_repeated_punctuation(self):
        assert clean_text("Wow!!!") == "wow!"
        assert clean_text("Really???") == "really?"
        assert clean_text("Hmm...") == "hmm."

    def test_strips_whitespace(self):
        assert clean_text("  hello   world  ") == "hello world"

    def test_handles_empty_string(self):
        assert clean_text("") == ""

    def test_handles_emoji_in_text(self):
        # Emojis should survive cleaning (they're informative for sarcasm)
        result = clean_text("Oh great 🙄")
        assert "great" in result

    def test_combined_pipeline(self):
        raw = "  OH WOW!!! Check https://news.com @reporter    "
        result = clean_text(raw)
        assert result == "oh wow! check @user"


# ──────────────────────────────────────────────
#  Model architecture tests (no training needed)
# ──────────────────────────────────────────────

class TestBERTSarcasmClassifier:
    """Test BERTSarcasmClassifier without loading real weights."""

    @pytest.fixture
    def tiny_model(self):
        """Build the smallest possible BERT config for fast tests."""
        from transformers import BertConfig, BertModel
        import torch.nn as nn

        cfg = BertConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
        )
        model = BERTSarcasmClassifier.__new__(BERTSarcasmClassifier)
        model.bert = BertModel(cfg)
        model.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.LayerNorm(16),
            nn.Dropout(0.1),
            nn.Linear(16, 1),
        )
        return model

    def test_forward_returns_logits_shape(self, tiny_model):
        batch_size, seq_len = 4, 16
        input_ids      = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

        logits = tiny_model(input_ids, attention_mask)
        assert logits.shape == (batch_size, 1), f"Expected (4,1), got {logits.shape}"

    def test_forward_with_labels_returns_loss_and_logits(self, tiny_model):
        batch_size, seq_len = 4, 16
        input_ids      = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        labels         = torch.randint(0, 2, (batch_size,))

        loss, logits = tiny_model(input_ids, attention_mask, labels)
        assert loss.ndim == 0,                  "Loss should be a scalar"
        assert loss.item() > 0,                 "Loss should be positive"
        assert logits.shape == (batch_size, 1), "Logits shape mismatch"

    def test_get_trainable_params_is_positive(self, tiny_model):
        assert tiny_model.get_trainable_params() > 0


class TestBiLSTMClassifier:
    """Test BiLSTMClassifier architecture."""

    @pytest.fixture
    def model(self):
        return BiLSTMClassifier(vocab_size=500, embed_dim=32, lstm_units=16, dropout_prob=0.0)

    def test_forward_returns_logits_shape(self, model):
        batch_size, seq_len = 8, 20
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        logits = model(input_ids)
        assert logits.shape == (batch_size, 1)

    def test_forward_with_labels_returns_loss_and_logits(self, model):
        batch_size, seq_len = 8, 20
        input_ids = torch.randint(0, 500, (batch_size, seq_len))
        labels    = torch.randint(0, 2, (batch_size,))
        loss, logits = model(input_ids, labels=labels)
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_pretrained_embeddings_loaded(self):
        vocab_size, embed_dim = 100, 32
        pretrained = torch.randn(vocab_size, embed_dim)
        model = BiLSTMClassifier(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            lstm_units=16,
            dropout_prob=0.0,
            pretrained_embeddings=pretrained,
        )
        assert torch.allclose(model.embedding.weight, pretrained)


# ──────────────────────────────────────────────
#  Data pipeline tests
# ──────────────────────────────────────────────

class TestMakeSplits:
    """Test stratified splitting logic."""

    def test_split_sizes_sum_to_total(self):
        import pandas as pd
        from sarcasm.data import make_splits

        df = pd.DataFrame({
            "text":  [f"text {i}" for i in range(1000)],
            "label": [i % 2 for i in range(1000)],
        })
        train, val, test = make_splits(df, train_ratio=0.7, val_ratio=0.15)
        assert len(train) + len(val) + len(test) == 1000

    def test_splits_are_stratified(self):
        import pandas as pd
        from sarcasm.data import make_splits

        df = pd.DataFrame({
            "text":  [f"text {i}" for i in range(1000)],
            "label": [i % 2 for i in range(1000)],
        })
        train, val, test = make_splits(df)
        for split in [train, val, test]:
            ratio = split["label"].mean()
            assert abs(ratio - 0.5) < 0.05, f"Stratification failed: label ratio = {ratio}"

    def test_invalid_ratio_raises(self):
        import pandas as pd
        from sarcasm.data import make_splits

        df = pd.DataFrame({"text": ["a"], "label": [0]})
        with pytest.raises(AssertionError):
            make_splits(df, train_ratio=0.9, val_ratio=0.2)


# ──────────────────────────────────────────────
#  Predictor integration test (with mock model)
# ──────────────────────────────────────────────

class TestSarcasmPredictor:
    """Test the predictor wrapper with a mocked model."""

    @pytest.fixture
    def mock_predictor(self):
        from sarcasm.predict import SarcasmPredictor
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Mock model that always returns logit=2.0 (→ prob≈0.88, SARCASTIC)
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[2.0]])

        return SarcasmPredictor(
            model=mock_model,
            tokenizer=tokenizer,
            max_length=32,
            threshold=0.5,
        )

    def test_predict_returns_expected_keys(self, mock_predictor):
        result = mock_predictor.predict("Oh great, another meeting!")
        assert set(result.keys()) == {"text", "label", "confidence", "prob"}

    def test_predict_sarcastic(self, mock_predictor):
        result = mock_predictor.predict("Oh great, another meeting!")
        assert result["label"] == "SARCASTIC"
        assert result["prob"] > 0.5
        assert 0 < result["confidence"] <= 1

    def test_predict_batch_length(self, mock_predictor):
        mock_predictor.model.return_value = torch.tensor([[2.0], [2.0], [2.0]])
        texts = ["text one", "text two", "text three"]
        results = mock_predictor.predict_batch(texts)
        assert len(results) == 3

    def test_confidence_is_prob_for_sarcastic(self, mock_predictor):
        result = mock_predictor.predict("whatever")
        if result["label"] == "SARCASTIC":
            assert abs(result["confidence"] - result["prob"]) < 1e-4
        else:
            assert abs(result["confidence"] - (1 - result["prob"])) < 1e-4
