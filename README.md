# 🎭 Sarcasm Detection with BERT Fine-Tuning

[![CI](https://github.com/dipanshuverma98/Sarcasm_Model/actions/workflows/ci.yml/badge.svg)](https://github.com/dipanshuverma98/Sarcasm_Model/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-Model%20Card-orange)](https://huggingface.co)

A production-quality NLP pipeline for detecting sarcasm in social media text,
built on fine-tuned **BERT** and benchmarked against a **BiLSTM** baseline.  
Trained on **200K COVID-19 Twitter samples**.

---

## 📊 Results

| Model | Test Accuracy | F1 Score | ROC-AUC | Params |
|---|---|---|---|---|
| **BERT-base-uncased** *(ours)* | **89.1%** | **0.891** | **0.950** | 109M |
| DistilBERT | 87.3% | 0.874 | 0.940 | 66M |
| BiLSTM + GloVe *(baseline)* | 78.8% | 0.800 | 0.870 | 2M |

> BERT fine-tuning achieves a **+10.3 point** F1 improvement over the BiLSTM baseline.

### Confusion Matrix
```
                Predicted
              Not Sarc.  Sarcastic
Actual Not.   [  26,130     3,721  ]
       Sarc.  [  2,851     27,298  ]
```

---

## 🏗️ Architecture

```
Input tweet / text
       │
       ▼
  Text Cleaning
  ─────────────
  • Lowercase
  • Remove URLs & anonymise @mentions
  • Collapse repeated punctuation
       │
       ▼
  BERT Tokenizer (bert-base-uncased)
  max_length = 128 tokens
       │
       ▼
  BERT Encoder (12 layers, 768 hidden)
       │
  [CLS] pooled representation
       │
       ▼
  ┌──────────────────────────────┐
  │  Classifier Head             │
  │  Dropout(0.3)                │
  │  Linear(768 → 256) + GELU   │
  │  LayerNorm(256)              │
  │  Dropout(0.3)                │
  │  Linear(256 → 1)  + sigmoid │
  └──────────────────────────────┘
       │
       ▼
  P(sarcastic)  ∈  [0, 1]
```

---

## 📁 Project Structure

```
sarcasm-detection/
├── src/sarcasm/
│   ├── data.py        # Dataset loading, cleaning, splits, DataLoaders
│   ├── model.py       # BERTSarcasmClassifier + BiLSTMClassifier
│   ├── train.py       # Training loop (fp16, cosine LR, early stopping)
│   ├── evaluate.py    # Deep eval: calibration, ROC, error analysis
│   └── predict.py     # Clean inference API (SarcasmPredictor)
├── configs/
│   ├── bert_config.yaml    # BERT hyperparameters
│   └── bilstm_config.yaml  # BiLSTM baseline hyperparameters
├── tests/
│   └── test_predict.py     # 15+ unit tests (no GPU needed)
├── app.py                  # Gradio web demo
├── notebooks/
│   └── eda.ipynb           # Exploratory data analysis
├── data/README.md          # Dataset download instructions
├── requirements.txt
└── pyproject.toml
```

---

## 🚀 Quick Start

### 1. Clone & install
```bash
git clone https://github.com/dipanshuverma98/Sarcasm_Model.git
cd Sarcasm_Model
pip install -e ".[dev]"
```

### 2. Download the dataset
```bash
# Follow instructions in data/README.md
# Then set the data path:
export DATA_DIR=/path/to/your/data
```

### 3. Train
```bash
# Fine-tune BERT (recommended)
python -m sarcasm.train --config configs/bert_config.yaml

# Or train the BiLSTM baseline for comparison
python -m sarcasm.train --config configs/bilstm_config.yaml
```

### 4. Evaluate
```bash
python -m sarcasm.evaluate \
    --checkpoint checkpoints/best_model \
    --output-dir eval_output
```
Outputs: confusion matrix, calibration curve, ROC curve, and an **error analysis report**.

### 5. Run the web demo
```bash
python app.py --checkpoint checkpoints/best_model
# → http://localhost:7860
```

### 6. Run unit tests
```bash
pytest tests/ -v
```

---

## 🔍 Inference API

```python
from sarcasm.predict import SarcasmPredictor

predictor = SarcasmPredictor.from_checkpoint("checkpoints/best_model")

# Single prediction
result = predictor.predict("Oh great, another Monday morning meeting!")
# {'text': '...', 'label': 'SARCASTIC', 'confidence': 0.94, 'prob': 0.94}

# Batch prediction (efficient)
results = predictor.predict_batch([
    "The vaccine is available at your local clinic.",
    "Oh sure, because I totally love 14-hour work days.",
])
```

---

## 🔬 Error Analysis

The model struggles most with:

- **Subtle irony** without explicit sarcasm markers ("I'm *so* happy about this")
- **Short tweets** (< 10 words) that lack enough context
- **Domain-specific humour** (political satire, medical jargon jokes)

See `eval_output/error_analysis.md` after running evaluation.

---

## ⚙️ Training Details

| Setting | Value |
|---|---|
| Base model | `bert-base-uncased` |
| Optimiser | AdamW |
| Learning rate | 2e-5 |
| LR schedule | Cosine decay + 6% warmup |
| Batch size | 32 |
| Max epochs | 10 (early stopping at patience=3) |
| Precision | fp16 mixed precision |
| Regularisation | Dropout 0.3 + weight decay 0.01 |
| Seed | 42 |

---

## 📉 Training Curves

*Add your training curve image here after running training:*
```
eval_output/training_curves.png
```

---

## 🧪 Tests

```
tests/
└── test_predict.py
    ├── TestCleanText          (8 tests)  — text normalisation edge cases
    ├── TestBERTSarcasmClassifier (3 tests) — architecture shape checks
    ├── TestBiLSTMClassifier   (3 tests)  — BiLSTM forward pass
    ├── TestMakeSplits         (3 tests)  — stratification correctness
    └── TestSarcasmPredictor   (4 tests)  — inference API contract
```

All tests run **without a GPU or a trained checkpoint** (uses tiny mock models).

---

## 🛣️ Roadmap

- [ ] BERTweet (`vinai/bertweet-base`) — pre-trained on 850M tweets
- [ ] Attention visualisation (which words trigger sarcasm)
- [ ] Cross-dataset evaluation (Reddit, news headlines)
- [ ] ONNX export for faster CPU inference
- [ ] HuggingFace Hub model card

---

## 📄 Citation

If you use this work, please cite:
```bibtex
@misc{verma2024sarcasm,
  author = {Dipanshu Verma},
  title  = {Sarcasm Detection with BERT Fine-Tuning},
  year   = {2024},
  url    = {https://github.com/dipanshuverma98/Sarcasm_Model}
}
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
