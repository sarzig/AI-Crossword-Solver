"""
Author: Sarah

NOTES: xxx tbd this is half-baked, consider deleting her.
"""

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


# Download necessary resources

# Load my dataset
clues = pd.read_csv(r"data/nytcrosswords.csv", encoding='latin1')


# Apply the function
clues["Cluster"] = ''
clues["possessive long quote"] = False
clues["long quote with blank"] = False

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
clues["number non a-z or 1-9 characters in clue"] = clues["Clue"].apply(lambda x: sum(not re.match(r"[A-Za-z0-9]", c) for c in x) / len(x) if len(x) > 0 else 0)
clues["contains e.g."] = clues["Clue"].str.contains(r"\be\.g\.", case=False, na=False)
clues["contains etc."] = clues["Clue"].str.contains(r"\betc\.", case=False, na=False)

def contains_long_quote_with_blank(clue):
    matches = re.findall(r'"([^"]*)"', clue)  # Find all quoted substrings
    for match in matches:
        words = match.split()
        if len(words) >= 6 and "_" in match:  # Check for 6+ words and an underscore
            return True
    return False

clues["long quote with blank"] = clues["Clue"].apply(contains_long_quote_with_blank)

def contains_possessive_long_quote(clue):
    # Regex pattern to match: Possessive phrase before a long quote
    pattern = r"(\b(?:\w+\s)*\w+['’]s?)\s+\"([^\"]{20,})\""

    match = re.search(pattern, clue)
    if match:
        quote = match.group(2)  # Extract the quote
        words = quote.split()
        if len(words) >= 6:  # Ensure the quote has at least 6 words
            return True
    return False

clues["possessive long quote"] = clues["Clue"].apply(contains_possessive_long_quote)

clue = clues[(clues["possessive long quote"] == True) & (clues["long quote with blank"] == True)]
clues = clues.sort_values("possessive long quote", ascending=False).sort_values("long quote with blank", ascending=False)

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
filtered_clues.to_csv("clue_clusters.csv", index=False)
