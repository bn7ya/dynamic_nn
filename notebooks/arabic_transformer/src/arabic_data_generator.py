"""
Arabic Conversational Data Generator

Generates ~5,000 samples of Arabic conversational data across 8 categories:
- Greetings (~1000)
- Personal info (~1000)
- Time/Date (~500)
- Location (~500)
- Weather (~500)
- Calculation (~500)
- Translation (~500)
- Code-switching (~500)
"""

import random
import csv
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

from .arabic_word_banks import (
    ARABIC_GREETINGS,
    COMMON_QUESTIONS,
    TIME_DATE,
    WEATHER,
    BASIC_INFORMATION,
    CALCULATIONS,
    CODE_SWITCHING,
    NAMES,
    CITIES,
    PROFESSIONS,
    HOBBIES,
    DAYS,
    MONTHS,
    ARABIC_NUMBERS,
    WEATHER_CONDITIONS,
    COUNTRIES_CAPITALS,
    DIRECTIONS,
    AFFIRMATIVE_RESPONSES,
    NEGATIVE_RESPONSES,
    POLITE_PHRASES,
)


@dataclass
class ConversationSample:
    """Single conversation sample"""
    input_text: str
    output_text: str
    conversation_type: str
    language: str  # "arabic", "mixed", "english"


class ArabicConversationalGenerator:
    """
    Generates Arabic conversational training data.
    Targets ~5,000 samples across conversation categories.
    """

    CONVERSATION_TYPES = [
        "greeting",
        "personal_info",
        "time_date",
        "location",
        "weather",
        "calculation",
        "general_info",
        "code_switching",
    ]

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        self.samples: List[ConversationSample] = []

    def _number_to_arabic(self, num: int) -> str:
        """Convert number to Arabic word representation"""
        if num in ARABIC_NUMBERS:
            return ARABIC_NUMBERS[num]

        if num < 100:
            tens = (num // 10) * 10
            ones = num % 10
            if ones == 0:
                return ARABIC_NUMBERS.get(tens, str(num))
            return f"{ARABIC_NUMBERS.get(ones, str(ones))} و {ARABIC_NUMBERS.get(tens, str(tens))}"

        return str(num)

    def generate_greeting_sample(self) -> ConversationSample:
        """Generate greeting exchange"""
        category = random.choice(list(ARABIC_GREETINGS.keys()))
        greeting_pair = random.choice(ARABIC_GREETINGS[category])

        # Optionally add polite phrase
        if random.random() < 0.3:
            polite = random.choice(POLITE_PHRASES)
            input_text = f"{greeting_pair[0]} {polite}"
        else:
            input_text = greeting_pair[0]

        return ConversationSample(
            input_text=input_text,
            output_text=greeting_pair[1],
            conversation_type="greeting",
            language="arabic"
        )

    def generate_personal_info_sample(self) -> ConversationSample:
        """Generate personal information Q&A with template expansion"""
        category = random.choice(list(COMMON_QUESTIONS.keys()))
        qa_pair = random.choice(COMMON_QUESTIONS[category])

        input_text = qa_pair[0]
        output_text = qa_pair[1]

        # Expand templates
        output_text = output_text.replace("{name}", random.choice(NAMES))
        output_text = output_text.replace("{city}", random.choice(CITIES))
        output_text = output_text.replace("{age}", str(random.randint(18, 60)))
        output_text = output_text.replace("{profession}", random.choice(PROFESSIONS))
        output_text = output_text.replace("{hobby}", random.choice(HOBBIES))
        output_text = output_text.replace("{married_status}", random.choice(["نعم متزوج", "لا اعزب", "نعم متزوجة", "لا عزباء"]))
        output_text = output_text.replace("{children_status}", random.choice(["نعم عندي اطفال", "لا ليس عندي", "عندي طفلين", "عندي ثلاثة"]))

        return ConversationSample(
            input_text=input_text,
            output_text=output_text,
            conversation_type="personal_info",
            language="arabic"
        )

    def generate_time_date_sample(self) -> ConversationSample:
        """Generate time/date Q&A"""
        category = random.choice(list(TIME_DATE.keys()))
        qa_pair = random.choice(TIME_DATE[category])

        input_text = qa_pair[0]
        output_text = qa_pair[1]

        # Expand templates
        hour = random.randint(1, 12)
        minute = random.choice([0, 15, 30, 45])
        period = random.choice(["صباحا", "مساء"])

        if minute == 0:
            time_str = f"{self._number_to_arabic(hour)} {period}"
        else:
            time_str = f"{self._number_to_arabic(hour)} و {self._number_to_arabic(minute)} دقيقة {period}"

        output_text = output_text.replace("{hour}", time_str)
        output_text = output_text.replace("{day}", random.choice(list(DAYS.values())))
        output_text = output_text.replace("{date}", f"{random.randint(1, 28)} {random.choice(list(MONTHS.values()))}")
        output_text = output_text.replace("{month}", random.choice(list(MONTHS.values())))
        output_text = output_text.replace("{year}", str(random.randint(2024, 2026)))

        return ConversationSample(
            input_text=input_text,
            output_text=output_text,
            conversation_type="time_date",
            language="arabic"
        )

    def generate_location_sample(self) -> ConversationSample:
        """Generate location-based Q&A"""
        qa_templates = BASIC_INFORMATION["location"]
        qa_pair = random.choice(qa_templates)

        input_text = qa_pair[0]
        output_text = qa_pair[1]

        country = random.choice(list(COUNTRIES_CAPITALS.keys()))
        capital = COUNTRIES_CAPITALS[country]
        place = random.choice(CITIES)

        input_text = input_text.replace("{place}", place)
        input_text = input_text.replace("{country}", country)

        output_text = output_text.replace("{place}", place)
        output_text = output_text.replace("{location}", random.choice(["الشرق الاوسط", "الخليج العربي", "شمال افريقيا"]))
        output_text = output_text.replace("{country}", country)
        output_text = output_text.replace("{capital}", capital)
        output_text = output_text.replace("{continent}", random.choice(["اسيا", "افريقيا"]))
        output_text = output_text.replace("{direction}", random.choice(DIRECTIONS))

        return ConversationSample(
            input_text=input_text,
            output_text=output_text,
            conversation_type="location",
            language="arabic"
        )

    def generate_weather_sample(self) -> ConversationSample:
        """Generate weather Q&A"""
        category = random.choice(list(WEATHER.keys()))
        qa_pair = random.choice(WEATHER[category])

        input_text = qa_pair[0]
        output_text = qa_pair[1]

        weather_condition = random.choice(list(WEATHER_CONDITIONS.values()))
        temp = random.randint(15, 45)

        output_text = output_text.replace("{weather}", weather_condition)
        output_text = output_text.replace("{temp}", str(temp))
        output_text = output_text.replace("{temp_desc}", random.choice(["حار جدا", "بارد", "معتدل", "دافئ"]))
        output_text = output_text.replace("{rain_response}", random.choice(["نعم ستمطر", "لا لن تمطر", "ربما", "الاحتمال قليل"]))
        output_text = output_text.replace("{forecast}", random.choice(["جميل", "غائم", "ممطر", "مشمس"]))
        output_text = output_text.replace("{storm_response}", random.choice(["لا", "نعم توقعات بعاصفة", "لا يوجد توقع"]))

        return ConversationSample(
            input_text=input_text,
            output_text=output_text,
            conversation_type="weather",
            language="arabic"
        )

    def generate_calculation_sample(self) -> ConversationSample:
        """Generate simple arithmetic in Arabic"""
        operation = random.choice(list(CALCULATIONS.keys()))
        template = random.choice(CALCULATIONS[operation])

        if operation == "addition":
            num1, num2 = random.randint(1, 50), random.randint(1, 50)
            result = num1 + num2
        elif operation == "subtraction":
            num1 = random.randint(20, 100)
            num2 = random.randint(1, num1 - 1)
            result = num1 - num2
        elif operation == "multiplication":
            num1, num2 = random.randint(1, 12), random.randint(1, 12)
            result = num1 * num2
        else:  # division
            num2 = random.randint(1, 10)
            result = random.randint(1, 10)
            num1 = num2 * result

        input_text = template.replace("{num1}", self._number_to_arabic(num1))
        input_text = input_text.replace("{num2}", self._number_to_arabic(num2))

        # Mix of Arabic number words and digits in response
        if random.random() < 0.5:
            output_text = f"الناتج هو {self._number_to_arabic(result)}"
        else:
            output_text = f"يساوي {result}"

        return ConversationSample(
            input_text=input_text,
            output_text=output_text,
            conversation_type="calculation",
            language="arabic"
        )

    def generate_general_info_sample(self) -> ConversationSample:
        """Generate general information Q&A"""
        qa_pair = random.choice(BASIC_INFORMATION["general"])

        return ConversationSample(
            input_text=qa_pair[0],
            output_text=qa_pair[1],
            conversation_type="general_info",
            language="arabic"
        )

    def generate_code_switching_sample(self) -> ConversationSample:
        """Generate mixed Arabic-English conversation"""
        pair = random.choice(CODE_SWITCHING)

        return ConversationSample(
            input_text=pair[0],
            output_text=pair[1],
            conversation_type="code_switching",
            language="mixed"
        )

    def generate_samples(
        self,
        num_samples: int = 5000,
        distribution: Optional[Dict[str, float]] = None
    ) -> List[ConversationSample]:
        """
        Generate balanced conversational dataset.

        Args:
            num_samples: Total number of samples to generate
            distribution: Optional dict mapping type -> fraction (must sum to 1.0)

        Returns:
            List of ConversationSample objects
        """
        if distribution is None:
            distribution = {
                "greeting": 0.20,
                "personal_info": 0.20,
                "time_date": 0.10,
                "location": 0.10,
                "weather": 0.10,
                "calculation": 0.10,
                "general_info": 0.10,
                "code_switching": 0.10,
            }

        generators = {
            "greeting": self.generate_greeting_sample,
            "personal_info": self.generate_personal_info_sample,
            "time_date": self.generate_time_date_sample,
            "location": self.generate_location_sample,
            "weather": self.generate_weather_sample,
            "calculation": self.generate_calculation_sample,
            "general_info": self.generate_general_info_sample,
            "code_switching": self.generate_code_switching_sample,
        }

        self.samples = []

        for conv_type, fraction in distribution.items():
            count = int(num_samples * fraction)
            generator = generators[conv_type]

            for _ in range(count):
                try:
                    sample = generator()
                    self.samples.append(sample)
                except Exception as e:
                    # Skip failed generations
                    continue

        # Shuffle samples
        random.shuffle(self.samples)

        return self.samples

    def get_statistics(self) -> Dict:
        """Get statistics about generated data"""
        if not self.samples:
            return {}

        type_counts = {}
        language_counts = {}
        unique_inputs = set()
        unique_outputs = set()

        for sample in self.samples:
            type_counts[sample.conversation_type] = type_counts.get(sample.conversation_type, 0) + 1
            language_counts[sample.language] = language_counts.get(sample.language, 0) + 1
            unique_inputs.add(sample.input_text)
            unique_outputs.add(sample.output_text)

        return {
            "total_samples": len(self.samples),
            "unique_inputs": len(unique_inputs),
            "unique_outputs": len(unique_outputs),
            "type_distribution": type_counts,
            "language_distribution": language_counts,
            "avg_input_length": sum(len(s.input_text) for s in self.samples) / len(self.samples),
            "avg_output_length": sum(len(s.output_text) for s in self.samples) / len(self.samples),
        }

    def save_to_csv(
        self,
        filepath: str,
        samples: Optional[List[ConversationSample]] = None
    ) -> None:
        """Save samples to CSV file"""
        if samples is None:
            samples = self.samples

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['input', 'output', 'type', 'language'])

            for sample in samples:
                writer.writerow([
                    sample.input_text,
                    sample.output_text,
                    sample.conversation_type,
                    sample.language
                ])

    def load_from_csv(self, filepath: str) -> List[ConversationSample]:
        """Load samples from CSV file"""
        samples = []

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                samples.append(ConversationSample(
                    input_text=row['input'],
                    output_text=row['output'],
                    conversation_type=row['type'],
                    language=row['language']
                ))

        self.samples = samples
        return samples

    def split_data(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[List[ConversationSample], List[ConversationSample], List[ConversationSample]]:
        """Split samples into train/val/test sets"""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        samples = self.samples.copy()
        random.shuffle(samples)

        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_samples = samples[:train_end]
        val_samples = samples[train_end:val_end]
        test_samples = samples[val_end:]

        return train_samples, val_samples, test_samples

    def save_splits(
        self,
        base_path: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Dict[str, str]:
        """Save train/val/test splits to separate CSV files"""
        train, val, test = self.split_data(train_ratio, val_ratio, test_ratio)

        base = Path(base_path)
        paths = {
            'train': str(base / 'train.csv'),
            'val': str(base / 'val.csv'),
            'test': str(base / 'test.csv'),
        }

        self.save_to_csv(paths['train'], train)
        self.save_to_csv(paths['val'], val)
        self.save_to_csv(paths['test'], test)

        return paths


def generate_dataset(
    num_samples: int = 5000,
    output_dir: str = None,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[List[ConversationSample], Dict]:
    """
    Convenience function to generate and save Arabic conversational dataset.

    Args:
        num_samples: Number of samples to generate
        output_dir: Directory to save CSV files (optional)
        seed: Random seed for reproducibility
        verbose: Print statistics

    Returns:
        Tuple of (samples, statistics)
    """
    generator = ArabicConversationalGenerator(seed=seed)
    samples = generator.generate_samples(num_samples)
    stats = generator.get_statistics()

    if verbose:
        print(f"Generated {stats['total_samples']} samples")
        print(f"Unique inputs: {stats['unique_inputs']}")
        print(f"Unique outputs: {stats['unique_outputs']}")
        print(f"\nType distribution:")
        for t, count in stats['type_distribution'].items():
            print(f"  {t}: {count} ({count/stats['total_samples']*100:.1f}%)")
        print(f"\nLanguage distribution:")
        for lang, count in stats['language_distribution'].items():
            print(f"  {lang}: {count} ({count/stats['total_samples']*100:.1f}%)")

    if output_dir:
        paths = generator.save_splits(output_dir)
        if verbose:
            print(f"\nSaved to:")
            for split, path in paths.items():
                print(f"  {split}: {path}")

    return samples, stats
