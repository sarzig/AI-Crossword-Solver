"""
Author: Sarah

This is the generic helpers file. 

Generally functions will be put here if they work well, do not have a better "home" 
elsewhere, and are polished.

Summary of functions-----------------------------------------------
General helpers:
* print_if(statement, print_bool)
* conditional_raise(error, raise_bool)

Get project specific resources
* def get_project_root()
* get_clues_dataframe(clues_path=None)

Text processing
* preprocess_lower_remove_punct_strip_whitespace(input_text)
* process_text_into_clue_answer(input_text)
-------------------------------------------------------------------
"""

import os
import re
import string
import pandas as pd
import hashlib

def print_if(statement, print_bool):
    """
    Helper to simplify conditional printing.

    :param statement: statement to print
    :param print_bool: boolean to print or not
    """
    if print_bool:
        print(statement)


def conditional_raise(error, raise_bool):
    """
    Helper to simplify the conditional raising of errors.

    :param error: error to raise
    :param raise_bool: boolean to raise
    """

    if raise_bool:
        raise error


def cool_error(error):
    """
    Print a very large error to hopefully convince the user to notice urgent
    action should be taken.
    :return: nothing
    """

    error_text = \
        """
    +-------------------------------------------------+
    |     .d88b.  888d888 888d888  .d88b.  888d888    |
    |    d8P  Y8b 888P"   888P"   d88""88b 888P"      |
    |    88888888 888     888     888  888 888        |
    |    Y8b.     888     888     Y88..88P 888        |
    |     "Y8888  888     888      "Y88P"  888        |
    +-------------------------------------------------+    
     """

    print(error_text)
    raise error


def get_clues_by_class(clue_class="all", classification_type="manual_only", prediction_threshold=0.8):
    """
    This queries two datasets:
      * nyt_crosswords.csv
      * Sarah's manually classified clues

    If classification_type is manual_only, then clues of the given clue class (or ALL classes with manual
    classes) will be returned in a df.

    If classification type is predicted_only, then the full kaggle dataset will be queried, with the
    predictions applied by my ML model. Beware, these are frequently incorrect, especially in pretty critical
    categories like "straight definition".

    :param: clue_class = if all, gives all clue types
    :param: classification_type= "manual_only", "predicted_only", "all"
    :return: df with columns ["Clue", "Word", "Class"]. Approximately 5k manually classed rows and 700k ML classed rows
    """

    loc = ""
    text =""

    # If only looking for manually classed clues, look in
    # the manually classified clues.xlsx
    if classification_type == "manual_only":
        text = "manual"
        loc = os.path.join(get_project_root(),
                           "data",
                           "manually classified clues.xlsx")

    # If looking for only predicted, then just use the full_clue set and assign predictions
    # Only get clues that have prediction threshold over 0.8
    if classification_type == "predicted_only":
        text = "ML"
        loc = os.path.join(get_project_root(),
                           "data",
                           "nytcrosswords_predicted_classes.xlsx")

    # read the dataframe from the location
    df = pd.read_excel(loc)
    class_series = (df["Class"]).dropna()
    class_series = class_series[class_series.apply(lambda x: isinstance(x, str))]
    classes = sorted(set(class_series))
    print(f"\nPulling {text} classified clues from\n{loc}.")

    # If class is not all, then subset to that class
    if clue_class == "all":
        print("Returning clues of all classes.\n")
    elif clue_class in classes:
        df = df[df["Class"] == clue_class]
    else:
        print("Unrecognized class. Please select a class from list:")
        classes = get_class_options()
        for each in classes:
            print(f"  * {each}")
        return None

    # Get only columns of interest
    columns_of_interest = ["Clue", "Word", "Class", "Confidence"]
    available_columns = [col for col in columns_of_interest if col in df.columns]
    df = df[available_columns].copy()

    # Make sure all columns in df are strings
    for col in df.columns:
        if col != "Confidence":
            df[col] = df[col].astype(str)

    return df


def get_class_options():
    """
    Simply looks into Sarah's manually classed clues and returns a list of all classes.

    :return: list of classes
    """
    manual_clues = get_clues_by_class(clue_class="all", classification_type="manual_only")
    unique_clues = list(set((manual_clues["Class"].to_list())))
    unique_clues.sort()
    return unique_clues


def get_vocab():
    """
    Fetch vocab from the combined_vocab.txt file.
    :return: set of the vocab
    """
    try:
        location = os.path.join(get_project_root(), "data", "combined_vocab.txt")
        with open(location, "r", encoding="utf-8") as f:
            print(f"Fetching combined vocab (nltk words, NYT data) from {location}")
            return set(line.strip() for line in f if line.strip())
    except Exception:
        location = os.path.join(get_project_root(), "combined_vocab.txt")
        with open(location, "r", encoding="utf-8") as f:
            print(f"Fetching combined vocab (nltk words, NYT data) from {location}")
            return set(line.strip() for line in f if line.strip())


def stable_hash(obj):
    """
    Stable hash will return the same random value every single time.

    :param obj: object
    :return: hashed integer
    """
    return int(hashlib.md5(str(obj).encode()).hexdigest(), 16)


def get_project_root():
    """
    Uses OS lib to search for cwd, and then walks back to project root.

    :return: os.path object
    """
    # get cwd and split into constituent parts
    cwd = os.getcwd()
    path_parts = cwd.split(os.sep)

    # Look for project name in the path
    project_root = ""
    if "ai_crossword_solver" in path_parts:
        index = path_parts.index("ai_crossword_solver")
        project_root = os.sep.join(path_parts[:index + 1])

    # If ai_crossword_solver isn't anywhere in the path, then flag
    else:
        error_text = "To our TA: Please note the parent project directory expects to be named 'ai_crossword_solver'"
        cool_error(FileNotFoundError(error_text))

    return project_root


def get_processed_puzzle_sample_root():
    """
    Quick helper to get the path to processed_puzzle_samples.
    :return: an os.path object
    """
    return os.path.join(get_project_root(),
                        "data",
                        "puzzle_samples",
                        "processed_puzzle_samples")


def get_clues_dataframe(clues_path=None, delete_dupes=False):
    """
    Uses OS lib to search for cwd, and then walks back to project root.

    Alternately, if you give it a path it just pulls from that.

    :return: the main Kaggle dataframe with all clues
    """

    if clues_path is None:
        # get cwd and split into constituent parts
        cwd = os.getcwd()
        path_parts = cwd.split(os.sep)

        # Look for project name in the path
        root = ""
        if "ai_crossword_solver" in path_parts:
            index = path_parts.index("ai_crossword_solver")
            root = os.sep.join(path_parts[:index + 1])

        # Load dataset
        clues_path = os.path.join(root, r"data", "nytcrosswords.csv")

    # Return the dataframe from that csv
    clues_df = pd.read_csv(clues_path, encoding='latin1')
    if delete_dupes:
        clues_df = clues_df.drop_duplicates(["Word", "Clue"])
    return clues_df


def get_100_most_common_clues_and_answers():
    """
    Every savvy cross-worder knows that "Actress Thurman" resolves a pesky puzzle triplet,
    that Jai Alai is a beautifully voweled Basque sport, and that Tae Kwon Do is a
    respected martial art.

    This function returns the 100 most common clues and answers, which we assume
    a person solving the crossword would know by good-old-fashioned rote memorization.

    # Ai assisted

    :return: dataframe with columns Clue, Word, count, and is_unique_clue
    """
    clues_df = get_clues_dataframe()

    # Top 200 most common Clue–Word pairs
    common_pairs = (
        clues_df.groupby(["Clue", "Word"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(200)
    )

    # All unique (Clue, Word) pairs in top clues
    filtered_clues_df = (
        clues_df[clues_df["Clue"].isin(common_pairs["Clue"])]
        .drop_duplicates(subset=["Clue", "Word"])
    )

    # Clues that only appear once among top pairs
    unique_clues_set = (
        filtered_clues_df["Clue"]
        .value_counts()
        .loc[lambda x: x == 1]
        .index
    )

    # Mark each Clue–Word pair in top 200 as unique or not
    common_pairs["is_unique_clue"] = common_pairs["Clue"].isin(unique_clues_set)
    common_pairs = common_pairs[common_pairs["is_unique_clue"] is True]  # only subset the clues we care about

    return common_pairs


def preprocess_lower_remove_punct_strip_whitespace(input_text):
    """
    Lowers case of input, replaces all white space and punctuation with " ".

    :param input_text: text to modify
    :return: new text
    """
    new_text = input_text.lower()

    # Remove punctuation that is NOT within a word (preserve in-word punctuation like "john's" -> "johns" and
    # "honky-tonk" -> "honkytonk")
    new_text = re.sub(r'\b(\w+)[\'-](\w+)\b', r'\1\2', new_text)  # Merge words with apostrophe or hyphen
    new_text = re.sub(fr"[{re.escape(string.punctuation)}]", " ", new_text)  # Remove other punctuation

    # Normalize whitespace
    new_text = re.sub(r'\s+', ' ', new_text).strip()

    return new_text


def process_text_into_clue_answer(input_text):
    """
    Removes all white space, converts characters into English equivalent.

    :param input_text: input_text to process into a clue answer
    :return: processed text
    """

    # Replace all possible whitespace in clue with nothing
    whitespace_regex = r"[\s\u00A0\u1680\u180E\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]"

    # These are all special characters which could theoretically
    # come up in the source pages of an answer. They need to be Anglicized into their closest
    # english approximation (a, e, i, o, u, y, n).
    replace_special_letters = {
        "a": ["á", "à", "â", "ä", "ã", "å", "ā", "ă", "ą", "ȧ", "ǎ"],
        "e": ["é", "è", "ê", "ë", "ē", "ĕ", "ė", "ę", "ě"],
        "i": ["í", "ì", "î", "ï", "ī", "ĭ", "į", "ı", "ȉ", "ȋ"],
        "o": ["ó", "ò", "ô", "ö", "õ", "ō", "ŏ", "ő", "ȯ", "ȱ", "ø"],
        "u": ["ú", "ù", "û", "ü", "ũ", "ū", "ŭ", "ů", "ű", "ų", "ȕ", "ȗ"],
        "y": ["ý", "ÿ", "ŷ", "ȳ", "ɏ"],
        "n": ["ñ", "ń", "ņ", "ň", "ŉ", "ŋ"]
    }

    # remove punctuation
    new_text = input_text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

    # Remove special letters by iterating across the dict.
    for base_letter, variants in replace_special_letters.items():
        for variant in variants:
            new_text = new_text.replace(variant, base_letter)

    # remove whitespace and lowercase it
    new_text = re.sub(whitespace_regex, "", new_text).upper()

    return new_text
