import os
import string
import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import Counter
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from nltk.corpus import words
from wordsegment import load, segment
from nltk import pos_tag


# Download NLTK libs
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('words')  # ENGLISH words
# Load a set of known English words
english_vocab = set(words.words())


def get_clues_dataframe():
    """
    Uses OS lib to search for cwd, and then walks back to project root.

    :return:
    """
    # get cwd and split into consitutent parts
    cwd = os.getcwd()
    path_parts = cwd.split(os.sep)

    # Look for project name in the path
    root = ""
    if "ai_crossword_solver" in path_parts:
        index = path_parts.index("ai_crossword_solver")
        root = os.sep.join(path_parts[:index + 1])

    # Load dataset
    clues_path = os.path.join(root, r"data//nytcrosswords.csv")

    clues_df = pd.read_csv(clues_path, encoding='latin1')
    return clues_df


def count_proper_nouns(text):
    """
    Counts the number of proper nouns (NNP, NNPS) in a given text.

    :param text: The input string to analyze
    :return: The count of proper nouns in the text
    """
    # Tokenize and POS tag
    words = word_tokenize(text)
    tagged_words = pos_tag(words)

    # proper nouns are either NNP or NNPS
    proper_noun_count = sum(1 for word, tag in tagged_words if tag in ['NNP', 'NNPS'])  # genai

    return proper_noun_count

from nltk.corpus import brown
from collections import Counter

# Download necessary NLTK datasets
nltk.download('brown')
nltk.download('averaged_perceptron_tagger')
brown_tagged_words = nltk.corpus.brown.tagged_words(tagset='universal')  # Using universal POS tags
word_tag_counts = Counter(brown_tagged_words)
nltk.download('universal_tagset')


def is_first_word_proper_noun(clue):
    """
    Uses word tokenization with nltk to assess if the first word of a sentence is
    a proper noun. There will occasionally be some ambiguity, but it's pretty
    good at giving the right answer with sufficient context.

    :param clue: clue to check, plaintext
    :return: True if first word is proper noun, false if it's not
    """

    tokenized_clue = word_tokenize(clue)

    # Lowercase only the first letter of the first word
    # Lowercase proper nouns can still be identified as proper nouns. By doing this
    # we increase the chance that ambiguous terms (like 'will' or 'mark') will
    # be classes as non-proper nouns, even when they ARE proper nouns
    #tokenized_clue[0] = tokenized_clue[0][0].lower() + tokenized_clue[0][1:]

    # If the first word is recognized
    pos_tagged_words = pos_tag(tokenized_clue)
    print(pos_tagged_words)

    # If the POS tag is Pronoun, plural pronoun, foreign word, or personal pronoun (Supposed to capture "I")
    # then return True. Otherwise, False!
    return pos_tagged_words[0][1] in ["NNP", "NNPS", "FW", "PRP"]


def uppercase_percentage(clue):
    """
    This function takes a clue, separates it into words, and counts the number
    of uppercase words, then divides that count by the total number of words.

    There can be ambiguity with POS. Some examples that flag as the first
    word NOT being a proper noun is: "Actress Moreno" or "Will Ferrel"

    :param clue: crossword clue, plain string
    :return: Percentage (float)
    """
    words = clue.split()  # splits at spaces

    upper_count = 0
    for i in range(len(words)):
        # strip out punctuation and check the first char
        stripped_word = words[i].lstrip(string.punctuation)

        # If there is some text left over after stripping, let's check further
        if stripped_word != "":

            # If the word is the first word, we need to inspect part of speech to know if it's
            # a proper noun, or if it's just beginning-of-sentence-capitalization
            if i == 0:
                print("i == 0 and stripped_word[0].isupper():")
                if stripped_word[0].isupper():
                    if is_first_word_proper_noun(clue):
                        upper_count += 1

            # Any words following first are easy - add 1 to upper count if they are uppercase
            else:
                if stripped_word[0].isupper():
                    upper_count += 1

    return upper_count / len(words)


def assign_primary_cluster(clue):
    if re.search(r"-above|-down", clue, re.IGNORECASE):
        return "Self-Referential"
    elif sum(c.isdigit() for c in clue) > len(clue) / 2:
        return "Mostly Numeric"
    elif len(re.sub(r"[A-Za-z]", "", clue)) == len(clue):
        return "Completely non-alphabet"
    elif re.fullmatch(r"[A-Za-z]+", clue):  # Checks if the clue is a single word with only letters
        return "Single Word"
    return "Other"  # Default category for unspecified cases


def analyze_pos_distribution(clue):
    tokens = word_tokenize(clue)
    pos_tags = pos_tag(tokens)
    pos_counts = Counter(tag for _, tag in pos_tags)
    total_words = len(tokens)
    pos_percentage = {pos: (count / total_words) * 100 for pos, count in pos_counts.items()}
    return pos_percentage


def classify_language(answer):
    if answer.istitle():  # Check if it's a proper noun (first letter capitalized)
        return "Proper Noun"
    elif answer.lower() in english_vocab:  # Check if it's in the English dictionary
        return "English"
    else:
        return "Foreign"



# Apply the function
#clues["Primary Cluster"] = clues["Clue"].apply(assign_primary_cluster)
#clues["Cluster"] = ''

def add_basic_features(clues_df):
    # add features
    clues_df["ends in question"] = clues_df["Clue"].str.endswith("?")
    clues_df["number words"] = clues_df["Clue"].str.split().apply(len)
    clues_df["length of clue"] = clues_df["Clue"].str.len()
    clues_df["no alphabet characters"] = clues_df["Clue"].apply(lambda x: len(re.sub(r'[A-Za-z]', '', x)))
    clues_df["is quote"] = clues_df["Clue"].str.endswith('"') & clues_df["Clue"].str.startswith('"')
    clues_df["contains underscore"] = clues_df["Clue"].str.contains(r"_", case=False, na=False)
    clues_df["contains asterisk"] = clues_df["Clue"].str.contains(r"\*", case=False, na=False)
    clues_df["contains done"] = clues_df["Clue"].str.contains(r"done", case=False, na=False)
    clues_df["number of non-consecutive periods in clue"] = clues_df["Clue"].apply(lambda x: len(re.findall(r"(?<!\.)\.(?!\.)", x)))
    clues_df["is ellipsis in clue"] = clues_df["Clue"].str.contains(r"\.\.\.", case=False, na=False)
    clues_df["avg word length"] = clues_df["Clue"].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()) if x.split() else 0)
    clues_df["number commas in clue"] = clues_df["Clue"].apply(lambda x: x.count(","))
    clues_df["percentage words (other than first word) that are upper-case"] = clues_df["Clue"].apply(uppercase_percentage)
    clues_df["number non a-z or 1-9 characters in clue"] = clues_df["Clue"].apply(lambda x: sum(not re.match(r"[A-Za-z0-9]", c) for c in x) / len(x) if len(x) > 0 else 0)
    clues_df["contains e.g."] = clues_df["Clue"].str.contains(r"\be\.g\.", case=False, na=False)
    clues_df["contains etc."] = clues_df["Clue"].str.contains(r"\betc\.", case=False, na=False)

    return clues_df

def kmeans_clustering_clues_dataframe(clues_df):
    # Drop non-feature columns
    features = clues_df.drop(columns=["Clue", "Date", "Word", "Primary Cluster", "Cluster"])

    # Identify binary columns (assumed to contain True/False values) and Convert True/False to 0/1
    binary_cols = [col for col in features.columns if clues_df[col].dtype == 'bool']
    features[binary_cols] = features[binary_cols].astype(int)

    # Identify numeric columns (excluding binary)
    numeric_cols = [col for col in features.columns if col not in binary_cols]

    # Standardize only numeric features using standardScaler
    scaler = StandardScaler()
    features[numeric_cols] = scaler.fit_transform(features[numeric_cols])

    # Apply KMeans clustering
    num_clusters = 30  # Change as needed
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clues_df["Cluster"] = kmeans.fit_predict(features)

    return clues_df

# Apply POS analysis to each clue
'''
print("here")
time1 = time.time()
pos_data = clues["Clue"].apply(analyze_pos_distribution).apply(pd.Series)
print("here2")
time2 = (time.time() - time1)
print(f"Time is {time2} seconds.")

# Merge POS data with original DataFrame
clues = pd.concat([clues, pos_data], axis=1).fillna(0)


clues = get_clues_dataframe()
new_clues = add_basic_features(clues)
new_clues = new_clues[new_clues['percentage words (other than first word) that are upper-case'] == 0]

Index(['Date', 'Word', 'Clue', 'ends in question', 'number words',
       'length of clue', 'no alphabet characters', 'is quote',
       'contains underscore', 'contains asterisk', 'contains done',
       'number of non-consecutive periods in clue', 'is ellipsis in clue',
       'avg word length', 'number commas in clue',
       'percentage words (other than first word) that are upper-case',
       'number non a-z or 1-9 characters in clue', 'contains e.g.',
       'contains etc.'],
      dtype='object')
'''