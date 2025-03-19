import os
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
    print("here")
    # get cwd and split into consitutent parts
    cwd = os.getcwd()
    path_parts = cwd.split(os.sep)
    print(cwd)
    # Look for project name in the path
    root = ""
    if "ai_crossword_solver" in path_parts:
        index = path_parts.index("ai_crossword_solver")
        root = os.sep.join(path_parts[:index + 1])
        print(root)

    # Load dataset
    clues_path = os.path.join(root, r"data/nytcrosswords.csv")

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


def uppercase_percentage(clue):
    """
    Given a clue, returns percentage of words which are uppercase (other
    than first character of the clue).

    :param clue:
    :return:
    """
    # split into words
    words = clue.split()
    total_words = len(words)

    print(words)
    if total_words == 0:
        return

    # ignore first word, and check if next words
    upper_count = 0
    for word in words[1:]:
        if word[0].isupper():
            upper_count += 1

    return upper_count / total_words


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
'''