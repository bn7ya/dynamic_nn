"""
Data generation module for text generation.
Generates bilingual (Arabic + English) word pattern data for sequence prediction.
"""

import csv
import random
import os
from typing import Tuple, List, Dict, Optional

from .word_banks import (
    ENGLISH_VOCAB, ARABIC_VOCAB,
    ASSOCIATIONS, ANALOGIES, SEQUENCES,
    TEMPLATE_PATTERNS, CATEGORY_COMPLETION
)


class GenerativeDataGenerator:
    """
    Generator for bilingual text generation training data.
    Creates input-output pairs for next-word prediction.
    """

    PATTERN_TYPES = [
        "association",  # color red -> apple
        "category",     # fruit apple banana -> orange
        "sequence",     # one two three -> four
        "template",     # the big house is -> old
        "analogy"       # king man woman -> queen
    ]

    def __init__(self, seed: int = 42):
        """
        Initialize the generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)

    def generate_association_pattern(self, language: str) -> Tuple[str, str]:
        """
        Generate a direct association pattern (color/category -> object).

        Args:
            language: 'en' or 'ar'

        Returns:
            Tuple of (input_text, output_text)
        """
        associations = ASSOCIATIONS[language]
        context = random.choice(list(associations.keys()))
        outputs = associations[context]
        output = random.choice(outputs)
        return context, output

    def generate_category_pattern(self, language: str) -> Tuple[str, str]:
        """
        Generate a category completion pattern (category + members -> next member).

        Args:
            language: 'en' or 'ar'

        Returns:
            Tuple of (input_text, output_text)
        """
        categories = CATEGORY_COMPLETION[language]
        category = random.choice(list(categories.keys()))
        members = categories[category].copy()
        random.shuffle(members)

        # Take 2-4 members as context, one as output
        num_context = random.randint(2, min(4, len(members) - 1))
        context_members = members[:num_context]
        output = members[num_context]

        input_text = f"{category} {' '.join(context_members)}"
        return input_text, output

    def generate_sequence_pattern(self, language: str) -> Tuple[str, str]:
        """
        Generate a number sequence pattern.

        Args:
            language: 'en' or 'ar'

        Returns:
            Tuple of (input_text, output_text)
        """
        sequences = SEQUENCES[language]
        context_tuple = random.choice(list(sequences.keys()))
        output = sequences[context_tuple]
        input_text = ' '.join(context_tuple)
        return input_text, output

    def generate_template_pattern(self, language: str) -> Tuple[str, str]:
        """
        Generate a template-based pattern (adjective context -> completion).

        Args:
            language: 'en' or 'ar'

        Returns:
            Tuple of (input_text, output_text)
        """
        templates = TEMPLATE_PATTERNS[language]
        template = random.choice(list(templates.keys()))
        completions = templates[template]
        noun = random.choice(list(completions.keys()))
        output = completions[noun]
        input_text = template.format(noun)
        return input_text, output

    def generate_analogy_pattern(self, language: str) -> Tuple[str, str]:
        """
        Generate an analogy pattern (A:B::C:?).

        Args:
            language: 'en' or 'ar'

        Returns:
            Tuple of (input_text, output_text)
        """
        analogies = ANALOGIES[language]
        context_tuple = random.choice(list(analogies.keys()))
        output = analogies[context_tuple]
        input_text = ' '.join(context_tuple)
        return input_text, output

    def generate_sample(
        self,
        pattern_type: str,
        language: str
    ) -> Tuple[str, str, str, str]:
        """
        Generate a single sample.

        Args:
            pattern_type: Type of pattern to generate
            language: 'en' or 'ar'

        Returns:
            Tuple of (input_text, output_text, pattern_type, language)
        """
        if pattern_type == "association":
            input_text, output_text = self.generate_association_pattern(language)
        elif pattern_type == "category":
            input_text, output_text = self.generate_category_pattern(language)
        elif pattern_type == "sequence":
            input_text, output_text = self.generate_sequence_pattern(language)
        elif pattern_type == "template":
            input_text, output_text = self.generate_template_pattern(language)
        elif pattern_type == "analogy":
            input_text, output_text = self.generate_analogy_pattern(language)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")

        return input_text, output_text, pattern_type, language

    def generate_samples(
        self,
        num_samples: int,
        balance: bool = True,
        en_ratio: float = 0.6
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Generate multiple samples.

        Args:
            num_samples: Total number of samples to generate
            balance: Whether to balance across pattern types
            en_ratio: Ratio of English samples (default 0.6 = 60%)

        Returns:
            Tuple of (input_texts, output_texts, pattern_types, languages)
        """
        input_texts = []
        output_texts = []
        pattern_types = []
        languages = []

        if balance:
            # Calculate samples per pattern type
            samples_per_pattern = num_samples // len(self.PATTERN_TYPES)

            for pattern_type in self.PATTERN_TYPES:
                # Calculate English vs Arabic for this pattern
                en_samples = int(samples_per_pattern * en_ratio)
                ar_samples = samples_per_pattern - en_samples

                # Generate English samples
                for _ in range(en_samples):
                    try:
                        inp, out, ptype, lang = self.generate_sample(pattern_type, "en")
                        input_texts.append(inp)
                        output_texts.append(out)
                        pattern_types.append(ptype)
                        languages.append(lang)
                    except (KeyError, IndexError):
                        # Skip if pattern type doesn't have enough data
                        pass

                # Generate Arabic samples
                for _ in range(ar_samples):
                    try:
                        inp, out, ptype, lang = self.generate_sample(pattern_type, "ar")
                        input_texts.append(inp)
                        output_texts.append(out)
                        pattern_types.append(ptype)
                        languages.append(lang)
                    except (KeyError, IndexError):
                        # Skip if pattern type doesn't have enough data
                        pass
        else:
            # Random generation
            for _ in range(num_samples):
                pattern_type = random.choice(self.PATTERN_TYPES)
                language = "en" if random.random() < en_ratio else "ar"
                try:
                    inp, out, ptype, lang = self.generate_sample(pattern_type, language)
                    input_texts.append(inp)
                    output_texts.append(out)
                    pattern_types.append(ptype)
                    languages.append(lang)
                except (KeyError, IndexError):
                    pass

        # Shuffle all lists together
        combined = list(zip(input_texts, output_texts, pattern_types, languages))
        random.shuffle(combined)

        if combined:
            input_texts, output_texts, pattern_types, languages = zip(*combined)
            return list(input_texts), list(output_texts), list(pattern_types), list(languages)
        return [], [], [], []

    def save_to_csv(
        self,
        input_texts: List[str],
        output_texts: List[str],
        pattern_types: List[str],
        languages: List[str],
        filepath: str
    ) -> None:
        """
        Save generated data to CSV file.

        Args:
            input_texts: List of input text samples
            output_texts: List of output text samples
            pattern_types: List of pattern type names
            languages: List of language codes
            filepath: Path to save CSV file
        """
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["input_text", "output_text", "pattern_type", "language"]
            )
            writer.writeheader()

            for inp, out, ptype, lang in zip(
                input_texts, output_texts, pattern_types, languages
            ):
                writer.writerow({
                    "input_text": inp,
                    "output_text": out,
                    "pattern_type": ptype,
                    "language": lang
                })

    def load_from_csv(
        self,
        filepath: str
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Load data from CSV file.

        Args:
            filepath: Path to CSV file

        Returns:
            Tuple of (input_texts, output_texts, pattern_types, languages)
        """
        input_texts = []
        output_texts = []
        pattern_types = []
        languages = []

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                input_texts.append(row['input_text'])
                output_texts.append(row['output_text'])
                pattern_types.append(row['pattern_type'])
                languages.append(row['language'])

        return input_texts, output_texts, pattern_types, languages


def generate_dataset(
    num_samples: int,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    en_ratio: float = 0.6
) -> Tuple[str, str, str]:
    """
    Generate a complete text generation dataset and save to CSV files.

    Args:
        num_samples: Total number of samples to generate
        output_dir: Directory to save CSV files
        train_ratio: Ratio of training samples (default 0.8)
        val_ratio: Ratio of validation samples (default 0.1)
        seed: Random seed for reproducibility
        en_ratio: Ratio of English samples (default 0.6)

    Returns:
        Tuple of (train_file_path, val_file_path, test_file_path)
    """
    generator = GenerativeDataGenerator(seed=seed)

    print(f"Generating {num_samples:,} samples...")
    input_texts, output_texts, pattern_types, languages = generator.generate_samples(
        num_samples,
        balance=True,
        en_ratio=en_ratio
    )

    actual_samples = len(input_texts)
    print(f"Generated {actual_samples:,} samples")

    # Split into train/val/test
    train_idx = int(actual_samples * train_ratio)
    val_idx = int(actual_samples * (train_ratio + val_ratio))

    train_data = (
        input_texts[:train_idx],
        output_texts[:train_idx],
        pattern_types[:train_idx],
        languages[:train_idx]
    )

    val_data = (
        input_texts[train_idx:val_idx],
        output_texts[train_idx:val_idx],
        pattern_types[train_idx:val_idx],
        languages[train_idx:val_idx]
    )

    test_data = (
        input_texts[val_idx:],
        output_texts[val_idx:],
        pattern_types[val_idx:],
        languages[val_idx:]
    )

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    train_file = os.path.join(output_dir, "train.csv")
    val_file = os.path.join(output_dir, "val.csv")
    test_file = os.path.join(output_dir, "test.csv")

    generator.save_to_csv(*train_data, train_file)
    generator.save_to_csv(*val_data, val_file)
    generator.save_to_csv(*test_data, test_file)

    print(f"Training samples: {len(train_data[0]):,} -> {train_file}")
    print(f"Validation samples: {len(val_data[0]):,} -> {val_file}")
    print(f"Test samples: {len(test_data[0]):,} -> {test_file}")

    return train_file, val_file, test_file


def get_data_statistics(
    input_texts: List[str],
    output_texts: List[str],
    pattern_types: List[str],
    languages: List[str]
) -> Dict:
    """
    Calculate statistics about the dataset.

    Args:
        input_texts: List of input texts
        output_texts: List of output texts
        pattern_types: List of pattern types
        languages: List of languages

    Returns:
        Dictionary with statistics
    """
    from collections import Counter

    pattern_counts = Counter(pattern_types)
    language_counts = Counter(languages)
    output_counts = Counter(output_texts)

    unique_inputs = len(set(input_texts))
    unique_outputs = len(set(output_texts))

    avg_input_len = sum(len(t.split()) for t in input_texts) / len(input_texts) if input_texts else 0

    return {
        "total_samples": len(input_texts),
        "unique_inputs": unique_inputs,
        "unique_outputs": unique_outputs,
        "avg_input_words": round(avg_input_len, 2),
        "pattern_distribution": dict(pattern_counts),
        "language_distribution": dict(language_counts),
        "top_outputs": output_counts.most_common(10),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate text generation dataset")
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--output", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--en-ratio", type=float, default=0.6)

    args = parser.parse_args()

    train_file, val_file, test_file = generate_dataset(
        args.samples,
        args.output,
        seed=args.seed,
        en_ratio=args.en_ratio
    )
