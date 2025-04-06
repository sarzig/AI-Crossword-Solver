import os
import re
import string

import pandas as pd

"""
This is the generic helpers file. 

Generally functions will be put here if they work well, do not have a better "home" 
elsewhere, and are polished.

Summary of functions-----------------------------------------------
-------------------------------------------------------------------
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


def get_project_root():
    """
    Uses OS lib to search for cwd, and then walks back to project root.

    :return:
    """
    # get cwd and split into constituent parts
    cwd = os.getcwd()
    path_parts = cwd.split(os.sep)

    # Look for project name in the path
    project_root = ""
    if "ai_crossword_solver" in path_parts:
        index = path_parts.index("ai_crossword_solver")
        project_root = os.sep.join(path_parts[:index + 1])

    return project_root


def get_clues_dataframe(clues_path=None):
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
        clues_path = os.path.join(root, r"data//nytcrosswords.csv")

    # Return the dataframe from that csv
    clues_df = pd.read_csv(clues_path, encoding='latin1')
    return clues_df


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
    new_text = re.sub(whitespace_regex, "", new_text).lower()

    return new_text
