"""
Word banks for bilingual text generation.
Contains English and Arabic vocabularies organized by category.
"""

# =============================================================================
# ENGLISH VOCABULARY
# =============================================================================

ENGLISH_VOCAB = {
    # Numbers (cardinal)
    "numbers": [
        "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten",
        "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen", "twenty"
    ],

    # Ordinals
    "ordinals": [
        "first", "second", "third", "fourth", "fifth",
        "sixth", "seventh", "eighth", "ninth", "tenth"
    ],

    # Colors
    "colors": [
        "red", "blue", "green", "yellow", "orange",
        "purple", "black", "white", "pink", "brown",
        "gray", "gold", "silver"
    ],

    # Animals
    "animals": [
        "cat", "dog", "bird", "fish", "lion",
        "tiger", "elephant", "horse", "cow", "sheep",
        "rabbit", "mouse", "bear", "wolf", "fox",
        "monkey", "snake", "eagle", "whale", "dolphin"
    ],

    # Fruits
    "fruits": [
        "apple", "banana", "orange", "grape", "mango",
        "strawberry", "cherry", "peach", "pear", "watermelon",
        "lemon", "lime", "coconut", "pineapple", "kiwi"
    ],

    # Vegetables
    "vegetables": [
        "carrot", "potato", "tomato", "onion", "garlic",
        "pepper", "cucumber", "lettuce", "spinach", "broccoli"
    ],

    # Size adjectives
    "sizes": [
        "big", "small", "large", "tiny", "huge",
        "tall", "short", "wide", "narrow", "long"
    ],

    # Quality adjectives
    "qualities": [
        "good", "bad", "new", "old", "fast",
        "slow", "hot", "cold", "warm", "cool",
        "bright", "dark", "loud", "quiet", "soft",
        "hard", "heavy", "light", "clean", "dirty"
    ],

    # Common nouns
    "nouns": [
        "house", "car", "tree", "book", "water",
        "food", "door", "window", "table", "chair",
        "bed", "phone", "computer", "road", "city",
        "mountain", "river", "ocean", "sky", "sun",
        "moon", "star", "cloud", "rain", "snow"
    ],

    # Verbs
    "verbs": [
        "run", "walk", "jump", "swim", "fly",
        "eat", "drink", "sleep", "wake", "think",
        "see", "hear", "feel", "touch", "smell"
    ],

    # Category markers
    "markers": [
        "color", "animal", "fruit", "number", "thing",
        "person", "place", "food", "vehicle", "plant"
    ],
}

# =============================================================================
# ARABIC VOCABULARY
# =============================================================================

ARABIC_VOCAB = {
    # Numbers (cardinal)
    "numbers": [
        "واحد", "اثنان", "ثلاثة", "اربعة", "خمسة",
        "ستة", "سبعة", "ثمانية", "تسعة", "عشرة",
        "احد عشر", "اثنا عشر", "ثلاثة عشر", "اربعة عشر", "خمسة عشر"
    ],

    # Ordinals
    "ordinals": [
        "اول", "ثاني", "ثالث", "رابع", "خامس",
        "سادس", "سابع", "ثامن", "تاسع", "عاشر"
    ],

    # Colors
    "colors": [
        "احمر", "ازرق", "اخضر", "اصفر", "برتقالي",
        "بنفسجي", "اسود", "ابيض", "وردي", "بني",
        "رمادي", "ذهبي", "فضي"
    ],

    # Animals
    "animals": [
        "قط", "كلب", "طائر", "سمكة", "اسد",
        "نمر", "فيل", "حصان", "بقرة", "خروف",
        "ارنب", "فار", "دب", "ذئب", "ثعلب",
        "قرد", "ثعبان", "نسر", "حوت", "دولفين"
    ],

    # Fruits
    "fruits": [
        "تفاحة", "موزة", "برتقالة", "عنب", "مانجو",
        "فراولة", "كرز", "خوخ", "اجاص", "بطيخ",
        "ليمون", "ليمون اخضر", "جوز الهند", "اناناس", "كيوي"
    ],

    # Vegetables
    "vegetables": [
        "جزر", "بطاطا", "طماطم", "بصل", "ثوم",
        "فلفل", "خيار", "خس", "سبانخ", "بروكلي"
    ],

    # Size adjectives
    "sizes": [
        "كبير", "صغير", "ضخم", "دقيق", "هائل",
        "طويل", "قصير", "واسع", "ضيق", "ممتد"
    ],

    # Quality adjectives
    "qualities": [
        "جيد", "سيء", "جديد", "قديم", "سريع",
        "بطيء", "حار", "بارد", "دافئ", "منعش",
        "مشرق", "مظلم", "صاخب", "هادئ", "ناعم",
        "صلب", "ثقيل", "خفيف", "نظيف", "متسخ"
    ],

    # Common nouns
    "nouns": [
        "بيت", "سيارة", "شجرة", "كتاب", "ماء",
        "طعام", "باب", "نافذة", "طاولة", "كرسي",
        "سرير", "هاتف", "حاسوب", "طريق", "مدينة",
        "جبل", "نهر", "محيط", "سماء", "شمس",
        "قمر", "نجمة", "سحابة", "مطر", "ثلج"
    ],

    # Verbs
    "verbs": [
        "يركض", "يمشي", "يقفز", "يسبح", "يطير",
        "ياكل", "يشرب", "ينام", "يستيقظ", "يفكر",
        "يرى", "يسمع", "يشعر", "يلمس", "يشم"
    ],

    # Category markers
    "markers": [
        "لون", "حيوان", "فاكهة", "رقم", "شيء",
        "شخص", "مكان", "طعام", "مركبة", "نبات"
    ],
}

# =============================================================================
# ASSOCIATIONS (Color/Category -> Object mappings)
# =============================================================================

ASSOCIATIONS = {
    "en": {
        # Color associations
        "color red": ["apple", "strawberry", "cherry", "tomato", "fire"],
        "color blue": ["sky", "ocean", "water", "blueberry", "ice"],
        "color green": ["grass", "leaf", "tree", "cucumber", "frog"],
        "color yellow": ["sun", "banana", "lemon", "corn", "gold"],
        "color orange": ["orange", "carrot", "sunset", "pumpkin", "tiger"],
        "color white": ["snow", "cloud", "milk", "cotton", "swan"],
        "color black": ["night", "crow", "coal", "shadow", "panther"],
        "color pink": ["flower", "flamingo", "pig", "rose", "candy"],
        "color brown": ["chocolate", "bear", "wood", "coffee", "earth"],
        "color purple": ["grape", "plum", "eggplant", "violet", "amethyst"],

        # Category associations
        "pet animal": ["dog", "cat", "rabbit", "fish", "bird"],
        "wild animal": ["lion", "tiger", "bear", "wolf", "elephant"],
        "sea animal": ["fish", "whale", "dolphin", "shark", "octopus"],
        "flying animal": ["bird", "eagle", "bat", "butterfly", "bee"],
        "farm animal": ["cow", "sheep", "horse", "pig", "chicken"],
    },
    "ar": {
        # Color associations (Arabic)
        "لون احمر": ["تفاحة", "فراولة", "كرز", "طماطم", "نار"],
        "لون ازرق": ["سماء", "محيط", "ماء", "توت", "جليد"],
        "لون اخضر": ["عشب", "ورقة", "شجرة", "خيار", "ضفدع"],
        "لون اصفر": ["شمس", "موزة", "ليمون", "ذرة", "ذهب"],
        "لون برتقالي": ["برتقالة", "جزر", "غروب", "يقطين", "نمر"],
        "لون ابيض": ["ثلج", "سحابة", "حليب", "قطن", "بجعة"],
        "لون اسود": ["ليل", "غراب", "فحم", "ظل", "نمر اسود"],
        "لون وردي": ["زهرة", "فلامنجو", "خنزير", "وردة", "حلوى"],
        "لون بني": ["شوكولاتة", "دب", "خشب", "قهوة", "تراب"],
        "لون بنفسجي": ["عنب", "برقوق", "باذنجان", "بنفسج", "جمشت"],

        # Category associations (Arabic)
        "حيوان اليف": ["كلب", "قط", "ارنب", "سمكة", "طائر"],
        "حيوان بري": ["اسد", "نمر", "دب", "ذئب", "فيل"],
        "حيوان بحري": ["سمكة", "حوت", "دولفين", "قرش", "اخطبوط"],
        "حيوان طائر": ["طائر", "نسر", "خفاش", "فراشة", "نحلة"],
        "حيوان مزرعة": ["بقرة", "خروف", "حصان", "خنزير", "دجاجة"],
    }
}

# =============================================================================
# NUMBER SEQUENCES
# =============================================================================

SEQUENCES = {
    "en": {
        # Cardinal number sequences
        ("one", "two", "three"): "four",
        ("two", "three", "four"): "five",
        ("three", "four", "five"): "six",
        ("four", "five", "six"): "seven",
        ("five", "six", "seven"): "eight",
        ("six", "seven", "eight"): "nine",
        ("seven", "eight", "nine"): "ten",
        ("eight", "nine", "ten"): "eleven",
        ("nine", "ten", "eleven"): "twelve",
        ("ten", "eleven", "twelve"): "thirteen",

        # Ordinal sequences
        ("first", "second", "third"): "fourth",
        ("second", "third", "fourth"): "fifth",
        ("third", "fourth", "fifth"): "sixth",
        ("fourth", "fifth", "sixth"): "seventh",
        ("fifth", "sixth", "seventh"): "eighth",
        ("sixth", "seventh", "eighth"): "ninth",
        ("seventh", "eighth", "ninth"): "tenth",

        # Even numbers
        ("two", "four", "six"): "eight",
        ("four", "six", "eight"): "ten",
        ("six", "eight", "ten"): "twelve",

        # Odd numbers
        ("one", "three", "five"): "seven",
        ("three", "five", "seven"): "nine",
        ("five", "seven", "nine"): "eleven",
    },
    "ar": {
        # Cardinal number sequences (Arabic)
        ("واحد", "اثنان", "ثلاثة"): "اربعة",
        ("اثنان", "ثلاثة", "اربعة"): "خمسة",
        ("ثلاثة", "اربعة", "خمسة"): "ستة",
        ("اربعة", "خمسة", "ستة"): "سبعة",
        ("خمسة", "ستة", "سبعة"): "ثمانية",
        ("ستة", "سبعة", "ثمانية"): "تسعة",
        ("سبعة", "ثمانية", "تسعة"): "عشرة",

        # Ordinal sequences (Arabic)
        ("اول", "ثاني", "ثالث"): "رابع",
        ("ثاني", "ثالث", "رابع"): "خامس",
        ("ثالث", "رابع", "خامس"): "سادس",
        ("رابع", "خامس", "سادس"): "سابع",
        ("خامس", "سادس", "سابع"): "ثامن",
        ("سادس", "سابع", "ثامن"): "تاسع",
        ("سابع", "ثامن", "تاسع"): "عاشر",
    }
}

# =============================================================================
# ANALOGIES (A:B :: C:?)
# =============================================================================

ANALOGIES = {
    "en": {
        # Gender pairs
        ("king", "man", "woman"): "queen",
        ("queen", "woman", "man"): "king",
        ("father", "man", "woman"): "mother",
        ("mother", "woman", "man"): "father",
        ("boy", "male", "female"): "girl",
        ("girl", "female", "male"): "boy",
        ("brother", "male", "female"): "sister",
        ("sister", "female", "male"): "brother",
        ("husband", "man", "woman"): "wife",
        ("wife", "woman", "man"): "husband",
        ("uncle", "man", "woman"): "aunt",
        ("aunt", "woman", "man"): "uncle",
        ("son", "male", "female"): "daughter",
        ("daughter", "female", "male"): "son",

        # Animal young
        ("dog", "puppy", "cat"): "kitten",
        ("cat", "kitten", "dog"): "puppy",
        ("horse", "foal", "cow"): "calf",
        ("cow", "calf", "horse"): "foal",
        ("sheep", "lamb", "pig"): "piglet",
        ("lion", "cub", "bear"): "cub",
        ("bird", "chick", "fish"): "fry",

        # Opposites
        ("hot", "fire", "ice"): "cold",
        ("cold", "ice", "fire"): "hot",
        ("big", "elephant", "mouse"): "small",
        ("small", "mouse", "elephant"): "big",
        ("fast", "cheetah", "turtle"): "slow",
        ("slow", "turtle", "cheetah"): "fast",
        ("day", "sun", "moon"): "night",
        ("night", "moon", "sun"): "day",
        ("up", "sky", "ground"): "down",
        ("down", "ground", "sky"): "up",

        # Seasonal
        ("hot", "summer", "cold"): "winter",
        ("cold", "winter", "hot"): "summer",
        ("spring", "flower", "leaf"): "autumn",

        # Size relationships
        ("large", "ocean", "pond"): "small",
        ("tall", "mountain", "hill"): "short",
    },
    "ar": {
        # Gender pairs (Arabic)
        ("ملك", "رجل", "امرأة"): "ملكة",
        ("ملكة", "امرأة", "رجل"): "ملك",
        ("اب", "رجل", "امرأة"): "ام",
        ("ام", "امرأة", "رجل"): "اب",
        ("ولد", "ذكر", "انثى"): "بنت",
        ("بنت", "انثى", "ذكر"): "ولد",
        ("اخ", "ذكر", "انثى"): "اخت",
        ("اخت", "انثى", "ذكر"): "اخ",
        ("زوج", "رجل", "امرأة"): "زوجة",
        ("زوجة", "امرأة", "رجل"): "زوج",
        ("عم", "رجل", "امرأة"): "عمة",
        ("عمة", "امرأة", "رجل"): "عم",
        ("ابن", "ذكر", "انثى"): "ابنة",
        ("ابنة", "انثى", "ذكر"): "ابن",

        # Animal young (Arabic)
        ("كلب", "جرو", "قط"): "هريرة",
        ("قط", "هريرة", "كلب"): "جرو",
        ("حصان", "مهر", "بقرة"): "عجل",
        ("بقرة", "عجل", "حصان"): "مهر",

        # Opposites (Arabic)
        ("حار", "نار", "جليد"): "بارد",
        ("بارد", "جليد", "نار"): "حار",
        ("كبير", "فيل", "فار"): "صغير",
        ("صغير", "فار", "فيل"): "كبير",
        ("سريع", "فهد", "سلحفاة"): "بطيء",
        ("بطيء", "سلحفاة", "فهد"): "سريع",
        ("نهار", "شمس", "قمر"): "ليل",
        ("ليل", "قمر", "شمس"): "نهار",

        # Seasonal (Arabic)
        ("حار", "صيف", "بارد"): "شتاء",
        ("بارد", "شتاء", "حار"): "صيف",
    }
}

# =============================================================================
# TEMPLATE PATTERNS (adjective-based completion)
# =============================================================================

TEMPLATE_PATTERNS = {
    "en": {
        # Size-based templates
        "the big {} is": {"house": "old", "car": "fast", "tree": "tall", "dog": "strong", "book": "heavy"},
        "the small {} is": {"house": "new", "car": "slow", "mouse": "quick", "bird": "cute", "box": "light"},
        "the tall {} is": {"man": "strong", "tree": "old", "building": "modern", "mountain": "cold"},
        "the short {} is": {"girl": "young", "path": "easy", "story": "funny", "break": "quick"},

        # Quality-based templates
        "the hot {} is": {"sun": "bright", "fire": "dangerous", "summer": "long", "coffee": "strong"},
        "the cold {} is": {"winter": "harsh", "ice": "hard", "wind": "strong", "water": "refreshing"},
        "the fast {} is": {"car": "red", "horse": "strong", "train": "long", "runner": "young"},
        "the slow {} is": {"turtle": "old", "traffic": "annoying", "progress": "steady"},
    },
    "ar": {
        # Size-based templates (Arabic)
        "ال{} الكبير هو": {"بيت": "قديم", "سيارة": "سريعة", "شجرة": "طويلة", "كلب": "قوي"},
        "ال{} الصغير هو": {"بيت": "جديد", "سيارة": "بطيئة", "فار": "سريع", "طائر": "لطيف"},

        # Quality-based templates (Arabic)
        "ال{} الحار هو": {"شمس": "ساطعة", "نار": "خطيرة", "صيف": "طويل"},
        "ال{} البارد هو": {"شتاء": "قاس", "جليد": "صلب", "ماء": "منعش"},
    }
}

# =============================================================================
# CATEGORY COMPLETION PATTERNS
# =============================================================================

CATEGORY_COMPLETION = {
    "en": {
        "fruit": ["apple", "banana", "orange", "grape", "mango", "strawberry", "cherry", "peach", "pear", "kiwi"],
        "animal": ["cat", "dog", "bird", "fish", "lion", "tiger", "elephant", "horse", "cow", "sheep"],
        "color": ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "pink", "brown"],
        "number": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
        "vehicle": ["car", "bus", "train", "plane", "boat", "bike", "truck", "motorcycle", "helicopter", "ship"],
        "body": ["hand", "foot", "head", "eye", "ear", "nose", "mouth", "arm", "leg", "finger"],
        "weather": ["sun", "rain", "snow", "wind", "cloud", "storm", "fog", "thunder", "lightning", "hail"],
        "food": ["bread", "rice", "meat", "fish", "egg", "cheese", "milk", "soup", "salad", "pasta"],
    },
    "ar": {
        "فاكهة": ["تفاحة", "موزة", "برتقالة", "عنب", "مانجو", "فراولة", "كرز", "خوخ", "اجاص", "كيوي"],
        "حيوان": ["قط", "كلب", "طائر", "سمكة", "اسد", "نمر", "فيل", "حصان", "بقرة", "خروف"],
        "لون": ["احمر", "ازرق", "اخضر", "اصفر", "برتقالي", "بنفسجي", "اسود", "ابيض", "وردي", "بني"],
        "رقم": ["واحد", "اثنان", "ثلاثة", "اربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة", "عشرة"],
        "مركبة": ["سيارة", "حافلة", "قطار", "طائرة", "قارب", "دراجة", "شاحنة", "دراجة نارية"],
        "طعام": ["خبز", "ارز", "لحم", "سمك", "بيض", "جبن", "حليب", "شوربة", "سلطة", "معكرونة"],
    }
}
