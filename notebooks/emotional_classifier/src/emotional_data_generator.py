"""
Emotional Text Data Generator

Generates synthetic emotional text samples for classification tasks.
Supports multiple emotions with bilingual (English/Arabic) text.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum


class Emotion(Enum):
    """Supported emotion categories."""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"
    NEUTRAL = "neutral"


@dataclass
class EmotionalSample:
    """Single emotional text sample."""
    text: str
    emotion: Emotion
    intensity: float  # 0.0 to 1.0
    language: str  # 'en' or 'ar'

    @property
    def label(self) -> int:
        """Get numeric label for emotion."""
        return list(Emotion).index(self.emotion)


class EmotionalDataGenerator:
    """
    Generates synthetic emotional text data for classification.

    Features:
    - 7 emotion categories (happy, sad, angry, fearful, surprised, disgusted, neutral)
    - Variable intensity levels
    - Bilingual support (English/Arabic)
    - Balanced or imbalanced class distribution
    """

    # English emotional templates
    ENGLISH_TEMPLATES = {
        Emotion.HAPPY: [
            "I feel so {intensity} today!",
            "This is {intensity} amazing news!",
            "I'm {intensity} thrilled about this!",
            "What a {intensity} wonderful day!",
            "I'm so {intensity} grateful for everything!",
            "This makes me {intensity} happy!",
            "I feel {intensity} blessed and joyful!",
            "Life is {intensity} beautiful!",
            "I can't stop smiling, I'm {intensity} happy!",
            "Everything is going {intensity} well!",
            "I'm {intensity} excited about the future!",
            "This is the {intensity} best day ever!",
            "I feel {intensity} alive and energetic!",
            "My heart is {intensity} full of joy!",
            "I'm {intensity} overjoyed right now!",
        ],
        Emotion.SAD: [
            "I feel so {intensity} down today.",
            "This news makes me {intensity} sad.",
            "I'm {intensity} heartbroken about this.",
            "Everything feels {intensity} hopeless.",
            "I can't stop feeling {intensity} miserable.",
            "This is {intensity} devastating news.",
            "I'm {intensity} depressed and lonely.",
            "Life feels {intensity} meaningless right now.",
            "I feel {intensity} empty inside.",
            "My heart is {intensity} heavy with sorrow.",
            "I'm {intensity} disappointed with how things turned out.",
            "I feel {intensity} lost and alone.",
            "This makes me {intensity} want to cry.",
            "I'm {intensity} grieving this loss.",
            "Everything seems {intensity} bleak.",
        ],
        Emotion.ANGRY: [
            "I'm so {intensity} angry right now!",
            "This makes me {intensity} furious!",
            "I can't believe how {intensity} unfair this is!",
            "I'm {intensity} outraged by this behavior!",
            "This is {intensity} unacceptable!",
            "I'm {intensity} livid about what happened!",
            "How dare they do this! I'm {intensity} mad!",
            "I feel {intensity} betrayed and angry!",
            "This is {intensity} infuriating!",
            "I'm {intensity} fed up with this situation!",
            "My blood is {intensity} boiling!",
            "I'm {intensity} frustrated beyond words!",
            "This makes me {intensity} want to scream!",
            "I'm {intensity} enraged by their actions!",
            "I feel {intensity} bitter about this!",
        ],
        Emotion.FEARFUL: [
            "I'm {intensity} scared of what might happen.",
            "This makes me feel {intensity} anxious.",
            "I'm {intensity} terrified right now.",
            "I feel {intensity} worried about the future.",
            "This situation is {intensity} frightening.",
            "I'm {intensity} afraid of the consequences.",
            "I feel {intensity} nervous and uneasy.",
            "This is {intensity} alarming news.",
            "I'm {intensity} dreading what comes next.",
            "I feel {intensity} panicked and scared.",
            "This makes me {intensity} uncomfortable.",
            "I'm {intensity} apprehensive about this.",
            "I feel {intensity} threatened by this situation.",
            "I'm {intensity} petrified with fear.",
            "This gives me {intensity} bad feelings.",
        ],
        Emotion.SURPRISED: [
            "I'm {intensity} shocked by this news!",
            "I can't believe it, I'm {intensity} amazed!",
            "This is {intensity} unexpected!",
            "I'm {intensity} stunned by what happened!",
            "Wow, this is {intensity} surprising!",
            "I'm {intensity} astonished by this!",
            "I never expected this, I'm {intensity} bewildered!",
            "This caught me {intensity} off guard!",
            "I'm {intensity} speechless right now!",
            "What a {intensity} twist of events!",
            "I'm {intensity} taken aback by this!",
            "This is {intensity} mind-blowing!",
            "I can't process this, I'm {intensity} dumbfounded!",
            "I'm {intensity} startled by the news!",
            "This is {intensity} unbelievable!",
        ],
        Emotion.DISGUSTED: [
            "I'm {intensity} disgusted by this.",
            "This makes me feel {intensity} sick.",
            "I find this {intensity} repulsive.",
            "This is {intensity} revolting behavior.",
            "I'm {intensity} appalled by what I see.",
            "This is {intensity} nauseating.",
            "I feel {intensity} repelled by this.",
            "This is {intensity} offensive to me.",
            "I'm {intensity} grossed out by this.",
            "This behavior is {intensity} despicable.",
            "I feel {intensity} contempt for this.",
            "This is {intensity} distasteful.",
            "I'm {intensity} sickened by this news.",
            "This is {intensity} vile and wrong.",
            "I find this {intensity} abhorrent.",
        ],
        Emotion.NEUTRAL: [
            "I don't have strong feelings about this.",
            "This is just a normal day.",
            "I'm neither happy nor sad about it.",
            "It's okay, nothing special.",
            "I feel indifferent about this matter.",
            "This doesn't really affect me.",
            "I have no particular opinion on this.",
            "It is what it is.",
            "I'm feeling balanced and calm.",
            "Nothing much is happening today.",
            "I'm in a stable emotional state.",
            "This is just routine for me.",
            "I don't feel strongly either way.",
            "Things are just average right now.",
            "I'm emotionally neutral about this.",
        ],
    }

    # Arabic emotional templates
    ARABIC_TEMPLATES = {
        Emotion.HAPPY: [
            "أشعر بسعادة {intensity} اليوم!",
            "هذا خبر {intensity} رائع!",
            "أنا {intensity} متحمس لهذا!",
            "يا له من يوم {intensity} جميل!",
            "أنا {intensity} ممتن لكل شيء!",
            "هذا يجعلني {intensity} سعيدا!",
            "أشعر بأنني {intensity} مبارك!",
            "الحياة {intensity} جميلة!",
            "لا أستطيع التوقف عن الابتسام!",
            "كل شيء يسير {intensity} بشكل جيد!",
        ],
        Emotion.SAD: [
            "أشعر بحزن {intensity} اليوم.",
            "هذا الخبر يجعلني {intensity} حزينا.",
            "قلبي {intensity} منكسر.",
            "كل شيء يبدو {intensity} ميؤوسا منه.",
            "لا أستطيع التوقف عن الشعور بالبؤس.",
            "هذا خبر {intensity} مدمر.",
            "أشعر بالوحدة والاكتئاب.",
            "الحياة تبدو {intensity} بلا معنى.",
            "أشعر بالفراغ في داخلي.",
            "قلبي {intensity} مثقل بالحزن.",
        ],
        Emotion.ANGRY: [
            "أنا {intensity} غاضب الآن!",
            "هذا يجعلني {intensity} محتدما!",
            "لا أصدق مدى {intensity} ظلم هذا!",
            "أنا {intensity} ثائر من هذا السلوك!",
            "هذا {intensity} غير مقبول!",
            "أنا {intensity} مستاء مما حدث!",
            "كيف يجرؤون على فعل هذا!",
            "أشعر بالخيانة والغضب!",
            "هذا {intensity} مثير للغضب!",
            "لقد سئمت من هذا الوضع!",
        ],
        Emotion.FEARFUL: [
            "أنا {intensity} خائف مما قد يحدث.",
            "هذا يجعلني أشعر بالقلق {intensity}.",
            "أنا {intensity} مرعوب الآن.",
            "أشعر بالقلق على المستقبل.",
            "هذا الوضع {intensity} مخيف.",
            "أنا خائف من العواقب.",
            "أشعر بالتوتر وعدم الارتياح.",
            "هذا خبر {intensity} مقلق.",
            "أخشى ما سيأتي بعد ذلك.",
            "أشعر بالذعر والخوف.",
        ],
        Emotion.SURPRISED: [
            "أنا {intensity} مصدوم من هذا الخبر!",
            "لا أصدق ذلك، أنا {intensity} مندهش!",
            "هذا {intensity} غير متوقع!",
            "أنا {intensity} مذهول مما حدث!",
            "واو، هذا {intensity} مفاجئ!",
            "أنا {intensity} مندهش من هذا!",
            "لم أتوقع هذا أبدا!",
            "هذا فاجأني {intensity}!",
            "أنا عاجز عن الكلام الآن!",
            "يا لها من مفاجأة {intensity}!",
        ],
        Emotion.DISGUSTED: [
            "أنا {intensity} مشمئز من هذا.",
            "هذا يجعلني أشعر بالغثيان.",
            "أجد هذا {intensity} مثيرا للاشمئزاز.",
            "هذا سلوك {intensity} مقزز.",
            "أنا {intensity} مستاء مما أرى.",
            "هذا {intensity} مقرف.",
            "أشعر بالنفور من هذا.",
            "هذا {intensity} مسيء لي.",
            "أنا {intensity} منزعج من هذا.",
            "هذا السلوك {intensity} حقير.",
        ],
        Emotion.NEUTRAL: [
            "ليس لدي مشاعر قوية حول هذا.",
            "هذا يوم عادي.",
            "لست سعيدا ولا حزينا.",
            "لا بأس، لا شيء مميز.",
            "أشعر بعدم المبالاة.",
            "هذا لا يؤثر علي حقا.",
            "ليس لدي رأي معين.",
            "هكذا هي الأمور.",
            "أشعر بالتوازن والهدوء.",
            "لا يحدث شيء كثير اليوم.",
        ],
    }

    # Intensity modifiers
    ENGLISH_INTENSITY = {
        "low": ["slightly", "somewhat", "a bit", "mildly"],
        "medium": ["quite", "pretty", "fairly", "rather"],
        "high": ["very", "extremely", "incredibly", "absolutely"],
    }

    ARABIC_INTENSITY = {
        "low": ["قليلا", "نوعا ما", "بعض الشيء"],
        "medium": ["جدا", "إلى حد ما", "بشكل ملحوظ"],
        "high": ["للغاية", "بشكل كبير", "تماما"],
    }

    def __init__(self, seed: int = 42):
        """Initialize generator with random seed."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.samples: List[EmotionalSample] = []

    def _get_intensity_modifier(self, intensity: float, language: str) -> str:
        """Get appropriate intensity modifier word."""
        modifiers = self.ENGLISH_INTENSITY if language == 'en' else self.ARABIC_INTENSITY

        if intensity < 0.33:
            level = "low"
        elif intensity < 0.67:
            level = "medium"
        else:
            level = "high"

        return self.rng.choice(modifiers[level])

    def _generate_sample(self, emotion: Emotion, language: str = 'en') -> EmotionalSample:
        """Generate a single emotional sample."""
        templates = self.ENGLISH_TEMPLATES if language == 'en' else self.ARABIC_TEMPLATES
        template = self.rng.choice(templates[emotion])

        # Generate intensity (skewed distribution for more realistic data)
        intensity = self.rng.beta(2, 2)  # Bell-shaped distribution

        modifier = self._get_intensity_modifier(intensity, language)
        text = template.format(intensity=modifier)

        return EmotionalSample(
            text=text,
            emotion=emotion,
            intensity=intensity,
            language=language
        )

    def generate_samples(
        self,
        num_samples: int = 5000,
        language_ratio: float = 0.7,  # Ratio of English samples
        balanced: bool = True,
        emotions: Optional[List[Emotion]] = None
    ) -> List[EmotionalSample]:
        """
        Generate emotional text samples.

        Args:
            num_samples: Total number of samples to generate
            language_ratio: Ratio of English to Arabic samples (0.0-1.0)
            balanced: If True, balance classes; if False, use natural distribution
            emotions: List of emotions to include (default: all)

        Returns:
            List of EmotionalSample objects
        """
        self.samples = []

        if emotions is None:
            emotions = list(Emotion)

        if balanced:
            # Equal samples per emotion
            samples_per_emotion = num_samples // len(emotions)
            remainder = num_samples % len(emotions)

            for i, emotion in enumerate(emotions):
                n = samples_per_emotion + (1 if i < remainder else 0)
                for _ in range(n):
                    lang = 'en' if self.rng.random() < language_ratio else 'ar'
                    self.samples.append(self._generate_sample(emotion, lang))
        else:
            # Natural distribution (neutral more common)
            emotion_probs = {
                Emotion.HAPPY: 0.15,
                Emotion.SAD: 0.12,
                Emotion.ANGRY: 0.10,
                Emotion.FEARFUL: 0.10,
                Emotion.SURPRISED: 0.08,
                Emotion.DISGUSTED: 0.05,
                Emotion.NEUTRAL: 0.40,
            }

            for _ in range(num_samples):
                emotion = self.rng.choice(
                    list(emotion_probs.keys()),
                    p=[emotion_probs[e] for e in emotion_probs.keys()]
                )
                lang = 'en' if self.rng.random() < language_ratio else 'ar'
                self.samples.append(self._generate_sample(emotion, lang))

        # Shuffle samples
        self.rng.shuffle(self.samples)
        return self.samples

    def split_data(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1
    ) -> Tuple[List[EmotionalSample], List[EmotionalSample], List[EmotionalSample]]:
        """Split samples into train/val/test sets."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

        n = len(self.samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        return (
            self.samples[:train_end],
            self.samples[train_end:val_end],
            self.samples[val_end:]
        )

    def get_statistics(self) -> Dict:
        """Get statistics about generated samples."""
        if not self.samples:
            return {}

        emotion_counts = {}
        language_counts = {'en': 0, 'ar': 0}
        intensities = []

        for sample in self.samples:
            emotion_counts[sample.emotion.value] = emotion_counts.get(sample.emotion.value, 0) + 1
            language_counts[sample.language] += 1
            intensities.append(sample.intensity)

        return {
            'total_samples': len(self.samples),
            'emotion_distribution': emotion_counts,
            'language_distribution': language_counts,
            'avg_intensity': np.mean(intensities),
            'intensity_std': np.std(intensities),
            'num_classes': len(set(s.emotion for s in self.samples)),
        }


def generate_emotional_dataset(
    num_samples: int = 5000,
    seed: int = 42,
    **kwargs
) -> Tuple[List[EmotionalSample], Dict]:
    """
    Convenience function to generate emotional dataset.

    Returns:
        Tuple of (samples, statistics)
    """
    generator = EmotionalDataGenerator(seed=seed)
    samples = generator.generate_samples(num_samples=num_samples, **kwargs)
    stats = generator.get_statistics()
    return samples, stats
