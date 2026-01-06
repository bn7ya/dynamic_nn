"""
Sentiment Analysis Project
Arabic + English Bilingual Sentiment Classification using Dynamic Neural Network
"""

from .data_generator import SentimentDataGenerator, generate_dataset
from .tokenizer import SimpleTokenizer
from .trainer import SentimentTrainer
from .evaluator import SentimentEvaluator
from .utils import setup_unicode_output, Timer

__version__ = "1.0.0"
__all__ = [
    "SentimentDataGenerator",
    "generate_dataset",
    "SimpleTokenizer",
    "SentimentTrainer",
    "SentimentEvaluator",
    "setup_unicode_output",
    "Timer",
]
