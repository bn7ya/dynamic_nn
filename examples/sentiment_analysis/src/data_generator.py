"""
Data generation module for Sentiment Analysis.
Generates bilingual (Arabic + English) sentiment data and saves to CSV.
"""

import csv
import random
import os
from typing import Tuple, List, Dict

from .word_banks import (
    WORD_BANKS, LABEL_MAP, SENTIMENTS, LANGUAGES
)


class SentimentDataGenerator:
    """
    Generator for bilingual sentiment analysis data.
    Creates synthetic sentences based on word banks.
    """

    # English sentence templates
    ENGLISH_TEMPLATES = [
        "{adverb} {adjective} {noun}",
        "{adjective} {noun} {phrase}",
        "I {verb} this {noun} {adverb} {adjective}",
        "This {noun} is {adverb} {adjective}",
        "{phrase} {adjective} {noun}",
        "The {noun} is {adjective} {phrase}",
        "{adverb} {adjective} {phrase}",
        "{adjective} {noun} I {verb} it",
        "This is {adverb} {adjective}",
        "{phrase} would {verb} this",
        "The {noun} {phrase}",
        "{adjective} {adjective} {noun}",
        "I {verb} the {adjective} {noun}",
        "{noun} is {adjective} {phrase}",
        "Really {adjective} {noun} {phrase}",
    ]

    # Arabic sentence templates
    ARABIC_TEMPLATES = [
        "{noun} {adjective} {adverb}",
        "{phrase} {adjective}",
        "{adjective} {noun} {phrase}",
        "{noun} {phrase}",
        "{adverb} {adjective} {noun}",
        "{phrase} {noun} {adjective}",
        "{adjective} {adverb} {phrase}",
        "{noun} {adjective} {phrase}",
    ]

    def __init__(self, seed: int = 42):
        """
        Initialize the generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)

    def generate_sentence(self, sentiment: str, language: str) -> str:
        """
        Generate a random sentence for given sentiment and language.

        Args:
            sentiment: One of 'positive', 'negative', 'neutral'
            language: One of 'en' (English) or 'ar' (Arabic)

        Returns:
            Generated sentence string
        """
        words = WORD_BANKS[sentiment][language]

        if language == "en":
            template = random.choice(self.ENGLISH_TEMPLATES)
        else:
            template = random.choice(self.ARABIC_TEMPLATES)

        sentence = template.format(
            adjective=random.choice(words["adjectives"]),
            noun=random.choice(words["nouns"]),
            verb=random.choice(words["verbs"]),
            adverb=random.choice(words["adverbs"]),
            phrase=random.choice(words["phrases"])
        )

        return sentence.strip()

    def generate_samples(
        self,
        num_samples: int,
        balance: bool = True
    ) -> Tuple[List[str], List[int], List[str], List[str]]:
        """
        Generate sentiment samples.

        Args:
            num_samples: Total number of samples to generate
            balance: Whether to balance across sentiments and languages

        Returns:
            Tuple of (texts, labels, sentiments, languages)
        """
        texts = []
        labels = []
        sentiment_list = []
        language_list = []

        if balance:
            samples_per_class = num_samples // len(SENTIMENTS)
            samples_per_language = samples_per_class // len(LANGUAGES)

            for sentiment in SENTIMENTS:
                for language in LANGUAGES:
                    for _ in range(samples_per_language):
                        text = self.generate_sentence(sentiment, language)
                        texts.append(text)
                        labels.append(LABEL_MAP[sentiment])
                        sentiment_list.append(sentiment)
                        language_list.append(language)
        else:
            for _ in range(num_samples):
                sentiment = random.choice(SENTIMENTS)
                language = random.choice(LANGUAGES)
                text = self.generate_sentence(sentiment, language)
                texts.append(text)
                labels.append(LABEL_MAP[sentiment])
                sentiment_list.append(sentiment)
                language_list.append(language)

        # Shuffle all lists together
        combined = list(zip(texts, labels, sentiment_list, language_list))
        random.shuffle(combined)
        texts, labels, sentiment_list, language_list = zip(*combined)

        return list(texts), list(labels), list(sentiment_list), list(language_list)

    def save_to_csv(
        self,
        texts: List[str],
        labels: List[int],
        sentiments: List[str],
        languages: List[str],
        filepath: str
    ) -> None:
        """
        Save generated data to CSV file.

        Args:
            texts: List of text samples
            labels: List of numeric labels
            sentiments: List of sentiment names
            languages: List of language codes
            filepath: Path to save CSV file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["text", "label", "sentiment", "language"]
            )
            writer.writeheader()

            for text, label, sentiment, language in zip(
                texts, labels, sentiments, languages
            ):
                writer.writerow({
                    "text": text,
                    "label": label,
                    "sentiment": sentiment,
                    "language": language
                })

    def load_from_csv(
        self,
        filepath: str
    ) -> Tuple[List[str], List[int], List[str], List[str]]:
        """
        Load data from CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            Tuple of (texts, labels, sentiments, languages)
        """
        texts = []
        labels = []
        sentiments = []
        languages = []

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(row['text'])
                labels.append(int(row['label']))
                sentiments.append(row['sentiment'])
                languages.append(row['language'])

        return texts, labels, sentiments, languages


def generate_dataset(
    num_samples: int,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[str, str]:
    """
    Generate a complete sentiment dataset and save to CSV files.

    Args:
        num_samples: Total number of samples to generate
        output_dir: Directory to save CSV files
        train_ratio: Ratio of training samples (default 0.8)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_file_path, test_file_path)
    """
    generator = SentimentDataGenerator(seed=seed)

    print(f"Generating {num_samples:,} samples...")
    texts, labels, sentiments, languages = generator.generate_samples(num_samples)

    # Split into train/test
    split_idx = int(len(texts) * train_ratio)

    train_texts = texts[:split_idx]
    train_labels = labels[:split_idx]
    train_sentiments = sentiments[:split_idx]
    train_languages = languages[:split_idx]

    test_texts = texts[split_idx:]
    test_labels = labels[split_idx:]
    test_sentiments = sentiments[split_idx:]
    test_languages = languages[split_idx:]

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, "train.csv")
    test_file = os.path.join(output_dir, "test.csv")

    generator.save_to_csv(
        train_texts, train_labels, train_sentiments, train_languages, train_file
    )
    generator.save_to_csv(
        test_texts, test_labels, test_sentiments, test_languages, test_file
    )

    print(f"Training samples: {len(train_texts):,} -> {train_file}")
    print(f"Test samples: {len(test_texts):,} -> {test_file}")

    return train_file, test_file


if __name__ == "__main__":
    # Test the generator
    import argparse

    parser = argparse.ArgumentParser(description="Generate sentiment dataset")
    parser.add_argument("--samples", type=int, default=120000)
    parser.add_argument("--output", type=str, default="../data")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train_file, test_file = generate_dataset(
        args.samples,
        args.output,
        seed=args.seed
    )
