# Emotional Classifier - Standard NN vs Dynamic NN Comparison
"""
This module compares Standard Neural Networks with Dynamic Neural Networks
for emotional text classification, showcasing the reward/penalty system.
"""

from .emotional_data_generator import EmotionalDataGenerator, EmotionalSample
from .text_vectorizer import TextVectorizer
from .standard_nn import StandardNeuralNetwork
from .dynamic_nn import DynamicNeuralNetwork
from .comparison import EmotionalClassifierComparison

__all__ = [
    'EmotionalDataGenerator',
    'EmotionalSample',
    'TextVectorizer',
    'StandardNeuralNetwork',
    'DynamicNeuralNetwork',
    'EmotionalClassifierComparison'
]
