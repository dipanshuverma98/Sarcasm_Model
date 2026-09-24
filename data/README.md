# Dataset

The model is trained on a COVID-19 Twitter sarcasm dataset.

## Download Instructions

### Option A — From Kaggle
1. Install the Kaggle CLI: `pip install kaggle`
2. Set up your API key: https://www.kaggle.com/docs/api
3. Download the dataset:
   ```bash
   kaggle datasets download -d ...  # replace with actual dataset slug
   unzip *.zip -d data/
   mv data/*.csv data/sarcasm_dataset.csv
   ```

### Option B — Manual
Download the CSV from the original source and place it at:
```
data/sarcasm_dataset.csv
```

## Expected Format

The CSV must have at least these two columns:

| Column | Type    | Description                          |
|--------|---------|--------------------------------------|
| text   | string  | Raw tweet or headline text           |
| label  | integer | 0 = Not sarcastic, 1 = Sarcastic     |

## Dataset Statistics

| Split | Samples | Sarcastic | Not Sarcastic |
|-------|---------|-----------|---------------|
| Train | 140,000 | 50%       | 50%           |
| Val   | 30,000  | 50%       | 50%           |
| Test  | 30,000  | 50%       | 50%           |

> **Note:** The dataset is balanced (50/50) using upsampling of the minority class before splitting.
