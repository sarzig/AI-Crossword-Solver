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
#from nltk.corpus import words
from wordsegment import load, segment
from nltk import pos_tag

from clue_classification_and_processing.helpers import get_clues_dataframe

# Download NLTK libs
'''
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('words')  # ENGLISH words
# Load a set of known English words
english_vocab = set(words.words())
'''

## Get data helpers ---------------------------------------------------------------------------

## String operation helpers ----------------------------------------------------------------------
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
    #print(pos_tagged_words)

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
                #print("i == 0 and stripped_word[0].isupper():")
                if stripped_word[0].isupper():
                    if is_first_word_proper_noun(clue):
                        upper_count += 1

            # Any words following first are easy - add 1 to upper count if they are uppercase
            else:
                if stripped_word[0].isupper():
                    upper_count += 1

    if len(words) == 0:
        return 0
    else:
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


def add_features(clues_df):
    """
    Given an input dataframe, this adds feature columns.

    Some clues were created or enhanced with genai.

    :param clues_df: input df
    :return: modified df with added columns
    """

    # Start by copying
    clues_df = clues_df.copy()

    # Ensure "Clue" is a string and fill NaNs
    if "Clue" in clues_df.columns:
        clues_df["Clue"] = clues_df["Clue"].fillna("").astype(str)

    # Length and casing related features
    clues_df["_f_number words"] = clues_df["Clue"].str.split().apply(len)
    clues_df["_f_length of clue"] = clues_df["Clue"].str.len()
    clues_df["_f_avg word length"] = clues_df["Clue"].apply(lambda x: sum(len(word) for word in x.split()) / len(x.split()) if x.split() else 0)
    clues_df["_f_percentage words that are upper-case"] = clues_df["Clue"].apply(uppercase_percentage)

    # Character related features
    clues_df["_f_ends in question"] = clues_df["Clue"].str.endswith("?")
    clues_df["_f_no alphabet characters"] = clues_df["Clue"].apply(lambda x: len(re.sub(r'[A-Za-z]', '', x)))
    clues_df["_f_is quote"] = clues_df["Clue"].str.endswith('"') & clues_df["Clue"].str.startswith('"')
    clues_df["_f_contains underscore"] = clues_df["Clue"].str.contains(r"_", case=False, na=False)
    clues_df["_f_contains asterisk"] = clues_df["Clue"].str.contains(r"\*", case=False, na=False)
    clues_df["_f_number of non-consecutive periods in clue"] = clues_df["Clue"].apply(lambda x: len(re.findall(r"(?<!\.)\.(?!\.)", x)))
    clues_df["_f_is ellipsis in clue"] = clues_df["Clue"].str.contains(r"...", case=False, na=False)
    clues_df["_f_number commas in clue"] = clues_df["Clue"].apply(lambda x: x.count(","))
    clues_df["_f_number non a-z or 1-9 characters in clue"] = clues_df["Clue"].apply(lambda x: sum(not re.match(r"[A-Za-z0-9]", c) for c in x) / len(x) if len(x) > 0 else 0)
    clues_df["_f_contains e.g."] = clues_df["Clue"].str.contains(r"\be\.g\.", case=False, na=False)
    clues_df["_f_contains etc."] = clues_df["Clue"].str.contains(r"\betc\.", case=False, na=False)
    clues_df["_f_contains in short"] = clues_df["Clue"].str.contains(r"\bin short\b", case=False, na=False)
    clues_df["_f_contains abbr"] = clues_df["Clue"].str.contains(r"Abbr.", case=False, na=False)
    clues_df["_f_contains amts."] = clues_df["Clue"].str.contains(r"amts.", case=False, na=False)
    clues_df["_f_contains briefly"] = clues_df["Clue"].str.contains(r"\bbriefly\b", case=False, na=False)
    clues_df["_f_contains dir."] = clues_df["Clue"].str.contains(" dir.", case=False, na=False)
    clues_df["_f_contains exclamation"] = clues_df["Clue"].str.contains('!"', case=False, na=False)
    clues_df["_f_starts with kind of"] = clues_df["Clue"].str.startswidth('kind of', case=False, na=False)
    clues_df["_f_contains it may be"] = clues_df["Clue"].str.contains('it may be', case=False, na=False)
    clues_df["_f_contains dir."] = clues_df["Clue"].str.contains(r"\bdir\.", case=False, na=False)
    clues_df["_f_contains bible clue"] = clues_df["Clue"].str.contains(
        r"\bbible\b|\bbiblical\b|\bjesus\b|old testament|new testament",
        case=False,
        na=False
    )
    clues_df["_f_contains ,maybe or ,perhaps"] = clues_df["Clue"].str.contains(r", (maybe|perhaps)", case=False, na=False)
    clues_df["_f_contains word before"] = clues_df["Clue"].str.contains(r"word before", case=False, na=False)

    # Need to add xxx tbd
    # Clue starts with "country where", "country that", "country that's", "country w", "city", "river to" , "river that", "river whose", "river of", river near", "river in", "river f"
    # river at,
    # More involved features
    clues_df = add_profession(clues_df)

    for col in clues_df.columns:
        if col.startswith("_f_"):
            clues_df[col] = pd.to_numeric(clues_df[col], errors="coerce")

    # Optional: fill any NaNs (caused by coercion)
    clues_df.fillna(0, inplace=True)

    return clues_df


def move_feature_columns_to_right_of_df(df):
    """
    Given a df with several columns, move columns with names starting with _f to the end of the column order.
    :param df: input dataframe.
    :return: input dataframe with columns rearranged (none deleted)
    """
    feature_cols = [col for col in df.columns if col.startswith("_f_")]
    other_cols = [col for col in df.columns if not col.startswith("_f_")]
    return df[other_cols + feature_cols]


def delete_feature_columns(df):
    """
    Given a df with several columns, delete columns with names starting with _f.
    :param df: input df
    :return: Input df with feature columsn deleted
    """
    feature_cols = [col for col in df.columns if col.startswith("_f_")]
    return df.drop(columns=feature_cols)


def select_numeric_features(df):
    """
    Select only the numeric features, which are columns starting with 'f_'.
    :param df: input DataFrame.
    :return: DataFrame with only numeric (feature) columns
    """
    numeric_cols = [col for col in df.columns if col.startswith("_f_")]

    return df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)


def add_profession(clues_df):
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
        "referee", "actor", "actress", "author", "poet", "novelist", "writer",
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
        "entrepreneur",
        "physicist",
        "physiologist",
        "adviser",
        "founder",
        "co-founder",
        "mogul",
        "tycoon",

        "lawman",
    ]

    pattern = r"\b(?:" + "|".join(prominent_professions) + r")\b"
    clues_df.loc[:, "_f_MentionsProfession"] = clues_df["Clue"].str.contains(pattern, flags=re.IGNORECASE, regex=True)
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
    clues_df.loc[:, "Cluster"] = kmeans.fit_predict(features)

    return clues_df

# Section where I play around with adding new features
'''
clues_df = get_clues_dataframe()
clues_df = add_features(clues_df)

column_of_interest = "_f_contains kind of"
clues_df[clues_df[column_of_interest] == True].to_csv(f"{column_of_interest}_subset.csv", index=False)

'''

