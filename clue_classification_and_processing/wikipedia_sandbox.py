import wikipediaapi
import re
import pandas as pd
import os

# Get the current working directory
cwd = os.getcwd()

# Initialize Wikipedia API with proper user-agent
wiki_wiki = wikipediaapi.Wikipedia(user_agent='MyProjectName (merlin@example.com)', language='en')

# Load dataset
clues_path = os.path.join(cwd, r"data\nytcrosswords.csv")
clues = pd.read_csv(clues_path, encoding='latin1')

# Add new columns for classification
clues["Cluster"] = ''
clues["possessive long quote"] = False
clues["long quote with blank"] = False


def extract_page_name(clue):
    """ Extracts the Wikipedia page name (everything before the first apostrophe) """
    return re.split(r"'", clue, maxsplit=1)[0] if "'" in clue else clue


def contains_long_quote_with_blank(clue):
    matches = re.findall(r'"([^"]*)"', clue)  # Find all quoted substrings
    for match in matches:
        words = match.split()
        if len(words) >= 6 and "_" in match:  # Check for 6+ words and an underscore
            return True
    return False



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


import re
import wikipediaapi

def find_blank_in_wiki(clue):
    """
    Extracts the page name, retrieves Wikipedia page content,
    and searches for the missing word(s) in a quote.
    """
    page_name = extract_page_name(clue)
    page_py = wiki_wiki.page(page_name)

    if not page_py.exists():
        print(f"Page '{page_name}' not found on Wikipedia.")
        return None

    # Extract the quote containing the blank
    match = re.search(r'"([^"]*___[^"]*)"', clue)
    if not match:
        print("No blank found in the quote.")
        return None

    quote_with_blank = match.group(1)  # Full quote including the blank
    quote_before_blank = quote_with_blank.replace("___", "").strip()  # Remove blank for search

    # Search for the quote in Wikipedia text
    wiki_text = page_py.text
    match_position = wiki_text.find(quote_before_blank)

    if match_position == -1:
        print(f"Quote '{quote_before_blank}' not found in Wikipedia page '{page_name}'.")
        return None

    # Extract surrounding words from Wikipedia text
    before_index = match_position
    after_index = match_position + len(quote_before_blank)

    # Get a few words before and after
    words_before = re.findall(r"\b\w+\b", wiki_text[max(0, before_index - 50):before_index])[-3:]  # Last 3 words before
    words_after = re.findall(r"\b\w+\b", wiki_text[after_index: after_index + 50])[:3]  # First 3 words after

    # Determine the missing word(s)
    missing_words = " ".join(words_before[-1:] + words_after[:1])  # Capture the word right before and after

    print(f"Extracted Missing Words: {missing_words}")
    return missing_words



# Apply feature extraction functions
clues["possessive long quote"] = clues["Clue"].apply(contains_possessive_long_quote)
clues["long quote with blank"] = clues["Clue"].apply(contains_long_quote_with_blank)

# Filter clues that match both conditions
filtered_clues = clues[(clues["possessive long quote"]) & (clues["long quote with blank"])]

# Iterate through filtered clues and find missing words in Wikipedia
for _, row in filtered_clues.iterrows():
    print("\n" + "-" * 80)
    print(f'Clue: {row["Clue"]}')

    page_name = extract_page_name(row["Clue"])
    print(f'Attempted Wikipedia Page Name: {page_name}')

    page_py = wiki_wiki.page(page_name)
    if page_py.exists():
        print(f'Wikipedia Summary:\n{page_py.summary[:300]}...')
    else:
        print("Wikipedia page not found.")

    ans = find_blank_in_wiki(row["Clue"])
    print(f"\n\nActual Answer: {row['Word']}")
    print(f'Predicted Answer: {ans}')
    input('')
