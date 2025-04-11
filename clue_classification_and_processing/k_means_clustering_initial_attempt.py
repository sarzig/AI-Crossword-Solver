"""
Author: Sarah
xxx tbd delete or keep some of cluster_attempt_outputs, maybe move to data to not clutter up this file
xxx tbd: I would love to ideally get this in an output which could be discussed in our
report. Doesn't need to be fancy, just a 2d or 3d space and a table to visualize
"yes, we put effort into this, and yes, this was of immense value to the early project,
even if it didn't end up being out primary pathway.
"""

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
from clue_classification_and_processing.clue_features import uppercase_percentage


# Download necessary resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

clues = pd.read_csv(r"data\nytcrosswords.csv", encoding='latin1')


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
clues["number non a-z or 1-9 characters in clue"] = clues["Clue"].apply(lambda x: sum(not re.match(r"[A-Za-z0-9 ]", c) for c in x) / len(x) if len(x) > 0 else 0)
clues["contains e.g."] = clues["Clue"].str.contains(r"\be\.g\.", case=False, na=False)
clues["contains etc."] = clues["Clue"].str.contains(r"\betc\.", case=False, na=False)
clues["quoted clues"] = clues["Clue"].apply(
    lambda x: isinstance(x, str) and x.count('"') == 2 and x.startswith('"') and x.endswith('"')
)
clues["contains dir."] = clues["Clue"].str.contains(r"\bdir\.", case=False, na=False)
clues["contains bible clue"] = clues["Clue"].str.contains(
    r"\bbible\b|\bbiblical\b|\bjesus\b|old testament|new testament",
    case=False,
    na=False
)
clues["contains ,maybe"] = clues["Clue"].str.contains(r", maybe", case=False, na=False)
clues["contains word before"] = clues["Clue"].str.contains(r"word before", case=False, na=False)

def is_roman_only(word):
    if type(word) ==str:
        return bool(re.fullmatch(r"[IVXLCDM]+", word.upper()))
    else:
        return ""
# Apply the function to create a new column
clues["OnlyRoman"] = clues["Word"].apply(is_roman_only)


# Need to add:
# Analogy : contains : twice and :: once
# Answer is shortening: includes "for short", "Abbr", in brief,  org., "text", network

vocab = [x.upper() for x in words.words()]
vocab_with_s = [f"{x}S" for x in vocab if not x.endswith("S")]
vocab = vocab + vocab_with_s
clues["in vocab"] = clues["Word"].isin(vocab)

# drop dupes
clues = clues.drop_duplicates(subset=["Word", "Clue"], keep="first").reset_index(drop=True)
subset_clues = clues.head(100000)
df_shuffled = subset_clues.sample(frac=1, random_state=42).reset_index(drop=True)
df_shuffled = df_shuffled.head(10000)

keywords = [
    "France", "French", "Paris",
    "Spain", "Spanish",
    "Mexico", "Tijuana", "Oaxaca", "Jalisco",
    "Rome", "Italian", "Italy"
]

clues["Word"] = clues["Word"].apply(lambda x: x if isinstance(x, str) else "")


# Create a regex pattern like: r"\b(france|french|paris|...)\b"
pattern = r"\b(" + "|".join(keywords) + r")\b"

clues["is analogy"] = clues["Clue"].str.contains(r"::", na=False)

# Flag clues that mention any of the keywords
clues["MentionsGeo"] = clues["Clue"].str.contains(pattern, flags=re.IGNORECASE, regex=True)

prominent_professions = [
    "author",
    "poet",
    "composer",
    "singer",
    "actor",
    "actress",
    "philanthropist",
    "ceo",
    "president",
    "mayor",
    "governor",
    "director",
    "producer",
    "dancer",
    "painter",
    "sculptor",
    "novelist",
    "editor",
    "journalist",
    "reporter",
    "host",
    "chef",
    "baker",
    "coach",
    "pilot",
    "surgeon",
    "doctor",
    "nurse",
    "scientist",
    "inventor",
    "engineer",
    "lawyer",
    "judge",
    "rabbi",
    "priest",
    "monk",
    "nun",
    "minister",
    "dean",
    "professor",
    "teacher",
    "student",
    "scholar",
    "critic",
    "curator",
    "violinist",
    "pianist",
    "guitarist",
    "drummer",
    "comedian",
    "clown",
    "magician",
    "bartender",
    "barista",
    "waiter",
    "waitress",
    "butler",
    "maid",
    "detective",
    "police",
    "officer",
    "firefighter",
    "soldier",
    "spy",
    "agent",
    "model",
    "designer",
    "tailor",
    "writer",
    "illustrator",
    "animator",
    "cartoonist",
    "blogger",
    "vlogger",
    "influencer",
    "athlete",
    "racer",
    "skater",
    "golfer",
    "boxer",
    "umpire",
    "referee",    "actor", "actress", "author", "poet", "novelist", "writer",
    "composer", "musician", "singer", "rapper", "pianist", "violinist",
    "artist", "painter", "sculptor", "director", "producer", "filmmaker",
    "comedian", "magician", "host", "broadcaster", "journalist", "editor",
    "blogger", "influencer", "chef", "designer", "model", "photographer",
    "philanthropist", "entrepreneur", "inventor", "engineer", "scientist",
    "astronaut", "explorer", "philosopher", "historian", "scholar", "professor",
    "teacher", "critic", "coach", "athlete", "boxer", "golfer", "racer",
    "skater", "runner", "cyclist", "swimmer", "surfer",
    "president", "prime minister", "governor", "mayor", "senator", "ambassador",
    "general", "admiral", "officer", "judge", "justice", "lawyer", "diplomat",
    "czar", "tsar", "monarch", "king", "queen", "emperor", "empress",
    "rabbi", "priest", "pastor", "imam", "monk", "nun", "bishop", "cardinal", "pope",
    "anchor",
    "newsman",
    "newscaster",
    "announcer",
    "emcee",
    "hostess",
    "broadcaster",
    "strategist",
    "consultant",
    "accountant",
    "economist",
    "banker",
    "trader",
    "broker",
    "entrepreneur"
]

pattern = r"\b(" + "|".join(prominent_professions) + r")\b"
clues["MentionsProfession"] = clues["Clue"].str.contains(pattern, flags=re.IGNORECASE, regex=True)


import spacy
import names
name_set = set(names.get_first_name(gender=None) for _ in range(10000))  # random sample


# Load English NER model (only load once)
nlp = spacy.load("en_core_web_sm")

def is_geography_clue_spacy(clue_text):
    doc = nlp(clue_text)
    return any(ent.label_ in {"GPE", "LOC"} for ent in doc.ents)
clues["geography entity"] = clues[clues["Clue"].apply(is_geography_clue_spacy)]

clues["is_first_name"] = clues["Word"].str.lower().isin(name.lower() for name in name_set)


clues[clues["contains word before"] == True].to_csv("word before clues.csv", index=False)


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

'''