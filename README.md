# Sarcasm Detection Model using BiLSTM

## Overview
This project implements a sarcasm detection model using a Bidirectional Long Short-Term Memory (BiLSTM) neural network. The model is trained to classify text as sarcastic or non-sarcastic based on labeled datasets..

## Features
- Uses BiLSTM for sequential text processing.
- Preprocesses text data using tokenization and padding.
- Trained on a sarcasm-labeled dataset.
- Outputs whether sarcasm or not

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Scikit-learn
- NLTK

## Dataset
The model is trained on a dataset containing sarcastic and non-sarcastic text samples. The dataset should be in JSON format with labeled text.

Example dataset format:
```
{
  "headline": "Text of the news headline",
  "is_sarcastic": 1
}
```

## Model Architecture
- **Embedding Layer**: Converts words into dense vectors.
- **BiLSTM Layer**: Captures the context of the text from both forward and backward directions.
- **Dense Layer**: Fully connected layer with activation function.
- **Output Layer**: Sigmoid activation for binary classification.

## Results
The model outputs whether text is sarcastic or not sarcastic.

