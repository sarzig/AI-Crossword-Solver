"""
Author: Sarah

Various functionality around filling in a blank is done here.

NOTES: xxx tbd: this is a half-baked file. I was trying to develop some ability to solve
fill-in-the-blank type clues.

"""
import re
from clue_classification_and_processing.helpers import preprocess_lower_remove_punct_strip_whitespace


def fill_in_the_blank_with_possible_source(clue, possible_source):
    """
    Given a clue containing a quote with one or more blanks (___),
    this function finds the best match in the possible source text and returns
    the missing words. Blanks can be at the beginning, middle, or end
    of a clue, and a blank can be repeated, in which case the SAME
    word in the source is searched for.

    xxx - need to change to allow for multiple words to be in the answer

    Function improved by genAI.

    :param clue: clue text containing a quote with multiple blanks
    :param possible_source: The text which may contain the full quote
    :return: List of words filling in the blanks or None if not found
    """

    # Remove the blanks from the quote and replace with placeholder. Then,
    # put the ___ back in.
    first_blank = "___"
    new_blank = "blankblankblank"

    # Clean the source text
    possible_source = preprocess_lower_remove_punct_strip_whitespace(possible_source)
    print(possible_source)

    # Extract clue.clue_text and replace the old style of blank with the new style
    clue = clue.replace(first_blank, new_blank)

    # Extract the quote containing one or more blanks (___)
    # If there is a match, capture it. xxx- this just uses first match right now. If there
    # are multiple quotes in a clue, this wouldn't get that.
    clue_quote_with_blank = None
    possible_match = re.search(rf'"([^"]*{new_blank}[^"]*)"', clue)
    if possible_match:
        clue_quote_with_blank = possible_match.group(1)

    # Remove all punctuation from the quote part of the clue, and then put the original blank back in
    clue_quote_with_blank = preprocess_lower_remove_punct_strip_whitespace(clue_quote_with_blank)
    clue_quote_with_blank = clue_quote_with_blank.replace(new_blank, first_blank)

    if not clue_quote_with_blank:
        return None  # No quote with a blank found

    # Convert the quote into a regex pattern with multiple wildcard captures for the blanks
    quote_pattern = re.escape(clue_quote_with_blank).replace(re.escape(first_blank), r'(\w+)')

    # Searching for a match in the possible source text
    match = re.search(pattern=quote_pattern, string=possible_source, flags=re.IGNORECASE)

    if match:
        groups = list(match.groups())
        # Check if the same word repeats across all blanks
        if len(set(groups)) == 1:
            return groups[0]  # Return the single word instead of multiple repeated words
        return groups  # Return all captured words as a list

    return None  # If no match is found, return None
