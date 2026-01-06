# Sentiment Analysis with Dynamic Neural Network

Bilingual (Arabic + English) sentiment classification using Dynamic Neural Network.

## Project Structure

```
sentiment_analysis/
├── main.py              # Main entry point
├── config.py            # Configuration settings
├── README.md            # This file
├── data/                # Generated CSV data files
│   ├── train.csv
│   └── test.csv
├── models/              # Saved model files
├── plots/               # Training visualizations
└── src/                 # Source modules
    ├── __init__.py
    ├── data_generator.py   # Data generation
    ├── tokenizer.py        # Text tokenization
    ├── trainer.py          # Model training
    ├── evaluator.py        # Model evaluation
    ├── word_banks.py       # Vocabulary banks
    └── utils.py            # Utility functions
```

## Quick Start

```bash
# Run with default settings (120k samples)
python main.py

# Generate custom number of samples
python main.py --samples 50000

# Use existing data (skip generation)
python main.py --skip-generate

# See all options
python main.py --help
```

## Features

- **Bilingual Data**: Arabic and English sentiment samples
- **Three Classes**: Positive, Negative, Neutral
- **Scalable**: Handles 100k+ samples efficiently
- **Configurable**: Vocabulary size, random seeds, etc.

## Module Usage

### Data Generation

```python
from src.data_generator import generate_dataset

train_file, test_file = generate_dataset(
    num_samples=100000,
    output_dir="./data",
    seed=42
)
```

### Tokenization

```python
from src.tokenizer import SimpleTokenizer, one_hot_encode

tokenizer = SimpleTokenizer(max_vocab_size=10000)
X = tokenizer.fit_transform_bow(texts)
y = one_hot_encode(labels, num_classes=3)
```

### Training

```python
from src.trainer import SentimentTrainer

trainer = SentimentTrainer(
    input_size=vocab_size,
    num_classes=3,
    seed=123
)
metrics = trainer.train(X_train, y_train)
```

### Evaluation

```python
from src.evaluator import SentimentEvaluator

evaluator = SentimentEvaluator()
metrics = evaluator.evaluate_from_trainer(trainer, X_test, y_test)
```

## Configuration

Edit `config.py` to customize:

- `DEFAULT_NUM_SAMPLES`: Number of samples to generate
- `MAX_VOCAB_SIZE`: Maximum vocabulary size
- `MODEL_SEED`: Random seed for reproducibility
- `COST_FUNCTION`: Loss function (CrossEntropy, MSE, etc.)
