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
from wordsegment import load, segment  # Library for word segmentation


# Download necessary resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load my dataset
clues = pd.read_csv(r"data/nytcrosswords.csv", encoding='latin1')

# Function to classify crossword answers
def classify_language(answer):
    # Attempt to segment words (e.g., ITSATRAP -> ["its", "a", "trap"])
    segmented_words = segment(answer.lower())  # Converts to lowercase before segmentation

    # If all words are capitalized (suggesting a proper noun), classify as Proper Noun
    if answer.isupper() and len(segmented_words) > 1:
        return "Proper Noun"

    # If all segmented words are in the English dictionary, classify as English
    if all(word in english_vocab for word in segmented_words):
        return "English"

    # If some words are in English and others are not, classify as Mixed
    if any(word in english_vocab for word in segmented_words):
        return "Mixed (English & Foreign)"

    # Otherwise, classify as Foreign
    return "Foreign"


# Define primary cluster categories
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

# Download the English words dataset
nltk.download('words')

# Load a set of known English words
english_vocab = set(words.words())
# Function to classify answers
def classify_language(answer):
    if answer.istitle():  # Check if it's a proper noun (first letter capitalized)
        return "Proper Noun"
    elif answer.lower() in english_vocab:  # Check if it's in the English dictionary
        return "English"
    else:
        return "Foreign"


# Apply the function
clues["Primary Cluster"] = clues["Clue"].apply(assign_primary_cluster)
clues["Cluster"] = ''

# add features
clues["ends in question"] = clues["Clue"].str.endswith("?")
clues["number words"] = clues["Clue"].str.split().apply(len)
clues["length of clue"] = clues["Clue"].str.len()
clues["no alphabet characters"] = clues["Clue"].apply(lambda x: len(re.sub(r'[A-Za-z]', '', x)))
clues["is quote"] = clues["Clue"].str.endswith('"') & clues["Clue"].str.startswith('"')
clues["contains underscore"] = clues["Clue"].str.contains(r"_", case=False, na=False)
clues["contains asterisk"] = clues["Clue"].str.contains(r"\*", case=False, na=False)
clues["contains done"] = clues["Clue"].str.contains(r"done", case=False, na=False)
clues["number of non-consecutive periods in clue"] = clues["Clue"].apply(lambda x: len(re.findall(r"(?<!\.)\.(?!\.)", x)))
clues["is ellipsis in clue"] = clues["Clue"].str.contains(r"\.\.\.", case=False, na=False)
clues["avg word length"] = clues["Clue"].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()) if x.split() else 0)
clues["number commas in clue"] = clues["Clue"].apply(lambda x: x.count(","))
clues["percentage words (other than first word) that are upper-case"] = clues["Clue"].apply(uppercase_percentage)
clues["number non a-z or 1-9 characters in clue"] = clues["Clue"].apply(lambda x: sum(not re.match(r"[A-Za-z0-9]", c) for c in x) / len(x) if len(x) > 0 else 0)
clues["contains e.g."] = clues["Clue"].str.contains(r"\be\.g\.", case=False, na=False)
clues["contains etc."] = clues["Clue"].str.contains(r"\betc\.", case=False, na=False)


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

# Drop non-feature columns
features = clues.drop(columns=["Clue", "Date", "Word", "Primary Cluster", "Cluster"])

# Identify binary columns (assumed to contain True/False values) and Convert True/False to 0/1
binary_cols = [col for col in features.columns if clues[col].dtype == 'bool']
features[binary_cols] = features[binary_cols].astype(int)

# Identify numeric columns (excluding binary)
numeric_cols = [col for col in features.columns if col not in binary_cols]

# Standardize only numeric features using standardScaler
scaler = StandardScaler()
features[numeric_cols] = scaler.fit_transform(features[numeric_cols])

# Apply KMeans clustering
num_clusters = 30  # Change as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
clues["Cluster"] = kmeans.fit_predict(features)

# print(df[["Clue", "Word", "cluster"]])
# Keep only the first 100 rows per cluster
filtered_clues = clues.groupby(["Cluster", "Primary Cluster"]).head(100)
filtered_clues = filtered_clues.sort_values("Cluster").sort_values("Primary Cluster")
filtered_clues = pd.concat([
    filtered_clues[filtered_clues["Primary Cluster"] != "Other"],  # Keep non-"Other" rows in original order
    filtered_clues[filtered_clues["Primary Cluster"] == "Other"].sort_values("Cluster")  # Sort "Other" rows by Cluster
], ignore_index=True)

# Export to CSV
clues.to_csv("all_clue_clusters.csv", index=False)
