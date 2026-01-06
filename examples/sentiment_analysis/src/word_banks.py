"""
Word banks for generating sentiment analysis data.
Contains English and Arabic vocabulary organized by sentiment.
"""

# =============================================================================
# English Word Banks
# =============================================================================

ENGLISH_POSITIVE = {
    "adjectives": [
        "amazing", "excellent", "fantastic", "wonderful", "great", "superb",
        "outstanding", "perfect", "brilliant", "incredible", "awesome", "terrific",
        "magnificent", "exceptional", "marvelous", "splendid", "delightful", "lovely",
        "fabulous", "remarkable", "impressive", "superior", "phenomenal", "stunning"
    ],
    "nouns": [
        "product", "service", "quality", "experience", "purchase", "item", "order",
        "delivery", "customer service", "value", "deal", "selection", "packaging"
    ],
    "verbs": [
        "love", "recommend", "enjoy", "appreciate", "adore", "cherish", "treasure"
    ],
    "adverbs": [
        "very", "extremely", "highly", "absolutely", "totally", "completely", "really",
        "incredibly", "exceptionally", "remarkably", "genuinely", "truly"
    ],
    "phrases": [
        "exceeded my expectations", "best purchase ever", "will buy again",
        "highly recommend", "worth every penny", "five stars", "top notch",
        "could not be happier", "pleasantly surprised", "blown away",
        "exactly what I needed", "perfect in every way", "must have",
        "game changer", "life saver", "best decision", "no regrets"
    ]
}

ENGLISH_NEGATIVE = {
    "adjectives": [
        "terrible", "awful", "horrible", "poor", "bad", "worst", "disappointing",
        "useless", "defective", "broken", "cheap", "low quality", "faulty",
        "pathetic", "dreadful", "appalling", "atrocious", "abysmal", "lousy",
        "inferior", "substandard", "shoddy", "rubbish", "garbage"
    ],
    "nouns": [
        "product", "service", "quality", "experience", "purchase", "item", "waste",
        "scam", "junk", "disaster", "nightmare", "disappointment", "ripoff"
    ],
    "verbs": [
        "hate", "regret", "avoid", "return", "complain", "warn", "despise"
    ],
    "adverbs": [
        "very", "extremely", "completely", "totally", "absolutely", "utterly",
        "incredibly", "horribly", "terribly", "awfully"
    ],
    "phrases": [
        "waste of money", "do not buy", "never again", "stay away",
        "complete disaster", "broke immediately", "does not work", "false advertising",
        "worst purchase ever", "total scam", "money down the drain", "buyer beware",
        "not worth it", "falling apart", "save your money", "huge disappointment"
    ]
}

ENGLISH_NEUTRAL = {
    "adjectives": [
        "okay", "average", "decent", "fair", "standard", "normal", "ordinary",
        "acceptable", "adequate", "moderate", "reasonable", "basic", "typical",
        "plain", "simple", "regular", "common", "usual", "mediocre", "passable"
    ],
    "nouns": [
        "product", "item", "purchase", "delivery", "service", "quality", "experience"
    ],
    "verbs": [
        "works", "functions", "serves", "does the job", "meets expectations"
    ],
    "adverbs": [
        "somewhat", "fairly", "reasonably", "moderately", "sufficiently"
    ],
    "phrases": [
        "nothing special", "as expected", "does the job", "no complaints",
        "what you pay for", "meets basic needs", "neither good nor bad",
        "could be better", "could be worse", "just okay", "not bad not great",
        "serves its purpose", "gets the job done", "middle of the road"
    ]
}

# =============================================================================
# Arabic Word Banks
# =============================================================================

ARABIC_POSITIVE = {
    "adjectives": [
        "رائع", "ممتاز", "مذهل", "جميل", "عظيم", "مثالي", "استثنائي",
        "فاخر", "راقي", "متميز", "بديع", "خيالي", "مبهر", "منقطع النظير"
    ],
    "nouns": [
        "منتج", "خدمة", "جودة", "تجربة", "شراء", "طلب", "توصيل", "سعر"
    ],
    "verbs": [
        "احب", "انصح", "اقدر", "اعشق", "افضل"
    ],
    "adverbs": [
        "جدا", "للغاية", "تماما", "حقا", "فعلا", "كثيرا"
    ],
    "phrases": [
        "افضل شراء", "يستحق كل ريال", "انصح به بشدة", "تجربة رائعة",
        "سأشتري مرة اخرى", "خمس نجوم", "فاق توقعاتي", "راضي تماما",
        "افضل منتج", "خدمة ممتازة", "جودة عالية", "سعر مناسب"
    ]
}

ARABIC_NEGATIVE = {
    "adjectives": [
        "سيء", "فظيع", "رديء", "مخيب", "فاشل", "معطل", "مكسور",
        "رخيص", "ضعيف", "كارثي", "مزعج", "محبط", "بائس", "مقرف"
    ],
    "nouns": [
        "منتج", "خدمة", "جودة", "تجربة", "شراء", "كارثة", "خسارة", "نصب"
    ],
    "verbs": [
        "اكره", "اندم", "تجنب", "ارجع", "اشتكي", "احذر"
    ],
    "adverbs": [
        "جدا", "للغاية", "تماما", "ابدا", "نهائيا"
    ],
    "phrases": [
        "اضاعة للمال", "لا تشتري", "لن اعود", "ابتعد عنه",
        "كارثة كاملة", "تعطل فورا", "لا يعمل", "غش وخداع",
        "اسوأ شراء", "نصب واحتيال", "خسرت فلوسي", "لا انصح به"
    ]
}

ARABIC_NEUTRAL = {
    "adjectives": [
        "عادي", "متوسط", "مقبول", "معقول", "بسيط", "طبيعي",
        "عادي جدا", "لا بأس", "محايد", "وسط"
    ],
    "nouns": [
        "منتج", "خدمة", "جودة", "تجربة", "شراء", "توصيل"
    ],
    "verbs": [
        "يعمل", "يؤدي الغرض", "مناسب", "كافي"
    ],
    "adverbs": [
        "نوعا ما", "الى حد ما", "بشكل معقول"
    ],
    "phrases": [
        "لا شيء مميز", "كما متوقع", "يؤدي الغرض", "لا شكاوى",
        "على قد الفلوس", "يلبي الحاجة", "لا سيء ولا ممتاز",
        "يمكن ان يكون افضل", "عادي جدا", "بدون مشاكل"
    ]
}

# =============================================================================
# Sentiment Configuration
# =============================================================================

WORD_BANKS = {
    "positive": {"en": ENGLISH_POSITIVE, "ar": ARABIC_POSITIVE},
    "negative": {"en": ENGLISH_NEGATIVE, "ar": ARABIC_NEGATIVE},
    "neutral": {"en": ENGLISH_NEUTRAL, "ar": ARABIC_NEUTRAL},
}

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

SENTIMENTS = ["positive", "negative", "neutral"]
LANGUAGES = ["en", "ar"]
