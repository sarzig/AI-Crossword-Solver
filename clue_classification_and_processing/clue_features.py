"""
Author: Sarah

This file is where I test new clues features. Its most important functions are:

Clue classification:
 * add_features(clues_df)
 * move_feature_columns_to_right_of_df(df)
 * delete_feature_columns(df)
 * select_numeric_features(df)

Helpers for clue classification:
 * count_proper_nouns

Clue k-means clustering:
 * kmeans_clustering_clues_dataframe(df) - completely functional clustering

"""

import re
import string
import pandas as pd
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from collections import Counter
from clue_classification_and_processing.helpers import get_clues_dataframe

'''
# Download NLTK libs - note that this section is not currently in use,
# but we may use it in the future!
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('words')  # ENGLISH words
# Load a set of known English words
english_vocab = set(words.words())
'''


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
    # tokenized_clue[0] = tokenized_clue[0][0].lower() + tokenized_clue[0][1:]

    # If the first word is recognized
    pos_tagged_words = pos_tag(tokenized_clue)
    # print(pos_tagged_words)

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
                # print("i == 0 and stripped_word[0].isupper():")
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


def add_profession(clues_df):
    """
    This is a simple lookup that helps tell if a clue contains a profession. Many clues are
    in the format "actress Biel", or "CEO Sundar" (which would have answers JESSICA and PICHAI.
    :param clues_df: the dataframe to which the new column _f_MentionsProfession will be added
    :return: the dataframe with that column added
    """
    prominent_professions = \
        ['accountant', 'actor', 'actress', 'admiral', 'adviser', 'agent', 'ambassador', 'anchor', 'animator',
         'artist', 'astronaut', 'athlete', 'author', 'baker', 'banker', 'barista', 'bartender', 'bishop', 'blogger',
         'boxer', 'broadcaster', 'broker', 'cardinal', 'cartoonist', 'ceo', 'chef', 'clown', 'co-founder', 'coach',
         'comedian', 'composer', 'consultant', 'critic', 'curator', 'cyclist', 'czar', 'dancer', 'dean', 'designer',
         'detective', 'diplomat', 'director', 'doctor', 'drummer', 'economist', 'editor', 'emcee', 'emperor', 'empress',
         'engineer', 'entrepreneur', 'explorer', 'filmmaker', 'firefighter', 'founder', 'general', 'golfer', 'governor',
         'guitarist', 'historian', 'host', 'hostess', 'illustrator', 'imam', 'influencer', 'inventor', 'journalist',
         'judge', 'justice', 'king', 'lawman', 'lawyer', 'magician', 'mayor', 'minister', 'model', 'mogul', 'monarch',
         'monk', 'musician', 'newscaster', 'newsman', 'novelist', 'nun', 'nurse', 'officer', 'painter', 'pastor',
         'philanthropist', 'philosopher', 'photographer', 'physicist', 'physiologist', 'pianist', 'pilot', 'poet',
         'police', 'pope', 'president', 'priest', 'prime minister', 'producer', 'professor', 'queen', 'rabbi', 'racer',
         'rapper', 'referee', 'reporter', 'runner', 'scholar', 'scientist', 'sculptor', 'senator', 'singer', 'skater',
         'soldier', 'spy', 'strategist', 'student', 'surfer', 'surgeon', 'swimmer', 'tailor', 'teacher', 'trader',
         'tsar', 'tycoon',  'suffragette', 'violinist', 'vlogger', 'writer']

    pattern = r"\b(?:" + "|".join(prominent_professions) + r")\b"
    clues_df.loc[:, "_f_MentionsProfession"] = clues_df["Clue"].str.contains(pattern, flags=re.IGNORECASE, regex=True)
    return clues_df


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
    clues_df["_f_avg word length"] = clues_df["Clue"].apply(
        lambda x: sum(len(word) for word in x.split()) / len(x.split()) if x.split() else 0)
    clues_df["_f_percentage words that are upper-case"] = clues_df["Clue"].apply(uppercase_percentage)

    # Character related features
    clues_df["_f_ends in question"] = clues_df["Clue"].str.endswith("?")
    clues_df["_f_no alphabet characters"] = clues_df["Clue"].apply(lambda x: len(re.sub(r'[A-Za-z]', '', x)))
    clues_df["_f_is quote"] = clues_df["Clue"].str.endswith('"') & clues_df["Clue"].str.startswith('"')
    clues_df["_f_contains underscore"] = clues_df["Clue"].str.contains(r"_", case=False, na=False)
    clues_df["_f_contains asterisk"] = clues_df["Clue"].str.contains(r"\*", case=False, na=False)
    clues_df["_f_number of non-consecutive periods in clue"] = clues_df["Clue"].apply(
        lambda x: len(re.findall(r"(?<!\.)\.(?!\.)", x)))
    clues_df["_f_is ellipsis in clue"] = clues_df["Clue"].str.contains(r"...", case=False, na=False)
    clues_df["_f_number commas in clue"] = clues_df["Clue"].apply(lambda x: x.count(","))
    clues_df["_f_number non a-z or 1-9 characters in clue"] = clues_df["Clue"].apply(
        lambda x: sum(not re.match(r"[A-Za-z0-9]", c) for c in x) / len(x) if len(x) > 0 else 0)

    # Contains some words which are short-cuts to figuring out the class
    clues_df["_f_contains e.g."] = clues_df["Clue"].str.contains(r"\be\.g\.", case=False, na=False)
    clues_df["_f_contains etc."] = clues_df["Clue"].str.contains(r"\betc\.", case=False, na=False)
    clues_df["_f_contains in short"] = clues_df["Clue"].str.contains(r"\bin short\b", case=False, na=False)
    clues_df["_f_contains abbr"] = clues_df["Clue"].str.contains(r"Abbr.", case=False, na=False)
    clues_df["_f_contains amts."] = clues_df["Clue"].str.contains(r"amts.", case=False, na=False)
    clues_df["_f_contains briefly"] = clues_df["Clue"].str.contains(r"\bbriefly\b", case=False, na=False)
    clues_df["_f_contains dir."] = clues_df["Clue"].str.contains(" dir.", case=False, na=False)
    clues_df["_f_contains exclamation"] = clues_df["Clue"].str.contains('!"', case=False, na=False)
    clues_df["_f_starts with kind of"] = clues_df["Clue"].str.lower().str.startswith('kind of')
    clues_df["_f_contains it may be"] = clues_df["Clue"].str.contains('it may be', case=False, na=False)
    clues_df["_f_contains dir."] = clues_df["Clue"].str.contains(r"\bdir\.", case=False, na=False)
    clues_df["_f_contains bible clue"] = clues_df["Clue"].str.contains(
        r"\bbible\b|\bbiblical\b|\bjesus\b|old testament|new testament",
        case=False,
        na=False
    )
    clues_df["_f_contains ,maybe or ,perhaps"] = clues_df["Clue"].str.contains(r", (?:maybe|perhaps)", case=False,
                                                                               na=False)
    clues_df["_f_contains word before"] = clues_df["Clue"].str.contains(r"word before", case=False, na=False)

    # Add more involved features
    clues_df = add_profession(clues_df)

    # This is required for success of random forest - all columns must be converted to numeric
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
    :param df: input df.
    :return: Input df with feature columns deleted
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


def kmeans_clustering_clues_dataframe(clues_df, n_clusters, feature_list=None):
    """
    Given a clues dataframe with features columns (denoted by a preceding "_f_"),
    perform k-means clustering on the dataframe, and sort by cluster.

    Example:
    my_clues_df = get_clues_dataframe()
    my_clues_df = add_features(my_clues_df)
    kmeans_res = kmeans_clustering_clues_dataframe(my_clues_df,
                                               n_clusters=3,
                                               feature_list=['_f_contains underscore', '_f_ends in question'])


    :param feature_list: the list of features to consider for clustering. Any feature not
                         in this list will be dropped prior to k-means clustering.
    :param n_clusters: number of clusters into which data should be separated
    :param clues_df: clues dataframe which has at least some feature columns
    :return:
    """
    if feature_list is None:
        # Drop any columns not starting with _f
        feature_cols = [col for col in clues_df.columns if col.startswith("_f")]
    else:
        feature_cols = [col for col in feature_list if col in clues_df.columns]

    # If no features are found, provide guidance and return original DataFrame
    if not feature_cols:
        print("No feature columns found. Please enrich the dataframe with features using:")
        print("    clues_df = add_features(clues_df)")
        return clues_df

    # Get a subset of the dataframe for which it is JUST features, we will later apply clustering
    # and re-apply this column to the dataframe.
    features = clues_df[feature_cols].copy()

    # Identify binary columns (assumed to contain True/False values) and Convert True/False to 0/1
    binary_cols = [col for col in features.columns if clues_df[col].dtype == 'bool']
    features[binary_cols] = features[binary_cols].astype(int)

    # Identify numeric columns (excluding binary)
    numeric_cols = [col for col in features.columns if col not in binary_cols]

    # Standardize only numeric features using standardScaler
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        features[numeric_cols] = scaler.fit_transform(features[numeric_cols])

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clues_df.loc[:, "Cluster"] = kmeans.fit_predict(features)
    clues_df = clues_df.sort_values("Cluster")
    clues_df = move_feature_columns_to_right_of_df(clues_df)

    return clues_df

'''
# Section where I play around with adding new features

# Here is where I play around with exporting certain features
my_clues_df = get_clues_dataframe()
my_clues_df = my_clues_df.head(30000)  # Lesser amount - good for troubleshooting
my_clues_df = add_features(my_clues_df)

column_of_interest = "_f_contains kind of"
# my_clues_df[my_clues_df[column_of_interest] == True].to_csv(f"{column_of_interest}_subset.csv", index=False)

# Here is where I perform k-means clustering.
kmeans_res = kmeans_clustering_clues_dataframe(my_clues_df,
                                               n_clusters=6,
                                               feature_list=['_f_contains underscore', '_f_ends in question'])

# xxx tbd - maybe add some plotting or example print outs of k-means clustering?
'''