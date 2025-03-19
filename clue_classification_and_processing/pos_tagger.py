from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download necessary resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# Sample sentence
sentence = "The quick brown fox jumps over the lazy dog."

# Tokenize and tag
tokens = word_tokenize(sentence)
pos_tags = pos_tag(tokens)

print(pos_tags)

def analyze_pos_distribution(clue):
    tokens = word_tokenize(clue)
    pos_tags = pos_tag(tokens)
    pos_counts = Counter(tag for _, tag in pos_tags)
    total_words = len(tokens)
    pos_percentage = {pos: (count / total_words) * 100 for pos, count in pos_counts.items()}
    return pos_percentage