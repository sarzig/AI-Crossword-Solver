"""
Various functionality around filling in a blank is done here.

"""
import re

from puzzle_objects.clue_and_board import Clue
import string


def preprocess_text(input_text):
    """
    Lowers case of input, replaces all white space and punctuation with " ".

    :param input_text: text to modify
    :return: new text
    """
    new_text = input_text.lower()
    new_text = re.sub(r'\s+', ' ', new_text)
    new_text = new_text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    new_text = re.sub(r'  +', ' ', new_text)
    return new_text.strip()


def process_text_into_clue_answer(text):
    """
    Removes all white space, converts characters
    :param text:
    :return:
    """
    # Replace all possible whitespace in clue with nothing
    whitespace_regex = r"[\s\u00A0\u1680\u180E\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]"

    # Define special letters which can come up in wikipedia and their a-z representation
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
    new_text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

    # Remove special letters
    for base_letter, variants in replace_special_letters.items():
        for variant in variants:
            new_text = new_text.replace(variant, base_letter)

    # remove whitespace
    new_text = re.sub(whitespace_regex, "", new_text)

    return new_text


def fill_in_the_blank_with_possible_source(clue: Clue, possible_source):
    """
    Given a clue containing a quote with one or more blanks (___),
    this function finds the best match in the possible source text and returns
    the missing words.

    :param clue: Clue object containing a quote with multiple blanks
    :param possible_source: The text which may contain the full quote
    :return: List of words filling in the blanks or None if not found
    """

    blank = "___"  # Define the blank format

    # Extract the quote containing one or more blanks (___)
    quote_match = re.search(rf'"([^"]*\b{blank}\b[^"]*)"', clue.clue_text)
    if not quote_match:
        return None  # No quote with a blank found

    quote_with_blanks = quote_match.group(1)  # Extract the quote with blanks

    # Convert the quote into a regex pattern with multiple wildcard captures for the blanks
    quote_pattern = re.escape(quote_with_blanks).replace(re.escape(blank), r'(\w+)')

    # Search for a match in the possible source text
    match = re.search(quote_pattern, possible_source, re.IGNORECASE)

    if match:
        groups = list(match.groups())
        # Check if the same word repeats across all blanks
        if len(set(groups)) == 1:
            return groups[0]  # Return the single word instead of multiple repeated words
        return groups  # Return all captured words as a list

    return None  # If no match is found, return None


b=fill_in_the_blank_with_possible_source(Clue('oh ho ho "hello ___ lady"'), "hello pretty lady")

clue_text = '"I could a tale unfold ___ lightest word / Would harrow up thy soul ...": "Hamlet"'
possible_source_text = """“I could a tale unfold whose lightest word
    Would harrow up thy soul, freeze thy young blood,
    Make thy two eyes like stars start from their spheres,
    Thy knotted and combined locks to part,
    And each particular hair to stand on end
    Like quills upon the fretful porpentine.
    But this eternal blazon must not be
    To ears of flesh and blood.
    List, list, O list!”
    """

a = fill_in_the_blank_with_possible_source(Clue(clue_text), possible_source_text)

