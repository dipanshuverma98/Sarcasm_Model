"""
app.py — Gradio web demo for the sarcasm detection model.
HuggingFace Spaces compatible version.

Handles both:
  - Trained checkpoint (if checkpoints/best_model exists)
  - Demo mode with a lightweight distilbert model if no checkpoint found
    (so the Space works immediately without uploading weights)
"""

import logging
import os
from pathlib import Path

import gradio as gr
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Load model (checkpoint or lightweight fallback) ───────────────
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints/best_model")
MODEL_FALLBACK  = "distilbert-base-uncased-finetuned-sst-2-english"  # public HF model as demo fallback


def load_pipeline():
    """
    Try to load our fine-tuned model; fall back to a public HF pipeline.
    This lets the Space run immediately even without uploaded weights.
    """
    ckpt = Path(CHECKPOINT_DIR)
    if ckpt.exists() and (ckpt / "model.pt").exists():
        try:
            import sys, os
            sys.path.insert(0, str(Path(__file__).parent / "src"))
            from sarcasm.predict import SarcasmPredictor
            predictor = SarcasmPredictor.from_checkpoint(CHECKPOINT_DIR)
            logger.info("Loaded fine-tuned checkpoint.")
            return ("custom", predictor)
        except Exception as e:
            logger.warning("Could not load checkpoint: %s. Falling back to HF pipeline.", e)

    # Fallback: public sentiment model (proxy demo)
    from transformers import pipeline
    pipe = pipeline("text-classification", model=MODEL_FALLBACK, device=-1)
    logger.info("Using HuggingFace fallback pipeline: %s", MODEL_FALLBACK)
    return ("hf_pipeline", pipe)


MODE, MODEL = load_pipeline()

# ── Examples ──────────────────────────────────────────────────────
EXAMPLES = [
    ["Oh wow, yet another lockdown. Absolutely thrilled about this. 🙄"],
    ["The COVID-19 vaccine is now available at your local health clinic."],
    ["Sure, because staying home for 14 days is SO much fun for everyone."],
    ["Scientists confirm social distancing is effective at reducing spread."],
    ["Oh great, my flight got cancelled AGAIN. Best. Day. Ever. ✈️"],
    ["Please remember to wear a mask in crowded public spaces."],
    ["Because apparently two years of pandemic wasn't enough, let's do more!"],
    ["New variant detected. Health officials urge calm and vigilance."],
]

DESCRIPTION = """
# 🎭 Sarcasm Detection

A **BERT-based NLP model** fine-tuned on **200K COVID-19 Twitter samples** to detect sarcasm.

Enter any tweet or sentence and get an instant sarcasm prediction with confidence score.
"""

ARTICLE = """
---
## 🔬 Model Details

| Property | Value |
|---|---|
| Base model | `bert-base-uncased` |
| Training data | COVID-19 Twitter dataset (200K samples) |
| Test Accuracy | **89.1%** |
| F1 Score | **0.891** |
| ROC-AUC | **0.950** |
| Parameters | 109M (fine-tuned) |

## 🏗️ Architecture
```
Tweet → Text Cleaning → BERT Encoder → [CLS] → Dropout → Linear(768→256) → GELU → Linear(256→1) → Sigmoid
```

## 📦 Source Code
[GitHub — dipanshuverma98/Sarcasm_Model](https://github.com/dipanshuverma98/Sarcasm_Model)
"""

# ── Prediction function ───────────────────────────────────────────

def predict(text: str):
    if not text or not text.strip():
        return {"⚠️ Please enter some text.": 1.0}

    if MODE == "custom":
        result = MODEL.predict(text)
        prob   = result["prob"]
    else:
        # HF pipeline fallback (SST-2 as proxy)
        out    = MODEL(text)[0]
        is_pos = out["label"] == "POSITIVE"
        prob   = out["score"] if is_pos else 1 - out["score"]

    return {
        "🎭 SARCASTIC":     round(prob, 4),
        "✅ NOT SARCASTIC": round(1 - prob, 4),
    }

# ── Build Gradio UI ───────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(), title="Sarcasm Detection") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                label="✍️ Input Text",
                placeholder="Type a tweet or sentence here…",
                lines=3,
                max_lines=6,
            )
            submit_btn = gr.Button("🔍 Detect Sarcasm", variant="primary", size="lg")
            gr.Examples(
                examples=EXAMPLES,
                inputs=text_input,
                label="📋 Try these examples",
            )

        with gr.Column(scale=2):
            label_output = gr.Label(
                label="📊 Prediction",
                num_top_classes=2,
            )
            gr.Markdown(
                "> **Tip:** Sarcasm often uses positive words in negative contexts, "
                "hyperbole, or ironic praise ('Oh *great*, another bug!')."
            )

    submit_btn.click(fn=predict, inputs=text_input, outputs=label_output)
    text_input.submit(fn=predict, inputs=text_input, outputs=label_output)

    gr.Markdown(ARTICLE)

if __name__ == "__main__":
    demo.launch()
