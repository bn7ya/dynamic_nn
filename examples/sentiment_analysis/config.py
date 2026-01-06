"""
Configuration settings for Sentiment Analysis project.
"""

import os

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

# =============================================================================
# Data Generation Settings
# =============================================================================

DEFAULT_NUM_SAMPLES = 120000
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# =============================================================================
# Tokenizer Settings
# =============================================================================

MAX_VOCAB_SIZE = 10000  # Limit vocabulary for memory efficiency
BATCH_LOG_INTERVAL = 10000

# =============================================================================
# Model Settings
# =============================================================================

NUM_CLASSES = 3
MODEL_SEED = 123
COST_FUNCTION = "CrossEntropy"
MODEL_NAME = "sentiment_classifier"

# =============================================================================
# Label Configuration
# =============================================================================

LABEL_MAP = {
    "positive": 2,
    "negative": 0,
    "neutral": 1
}

LABEL_NAMES = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}
