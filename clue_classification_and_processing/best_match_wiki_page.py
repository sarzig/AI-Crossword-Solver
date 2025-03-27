import os
import re
import string
import time
import openpyxl
import pandas as pd
import wikipediaapi
import spacy


# Load spaCy English model with Named Entity Recognition (NER)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def best_wikipedia_pages_for_clue(clue_text):
    """
    Given a clue text, this uses NLP to get a list of recommended wikipedia pages.

    :param clue_text:
    :return: list of page names, or None
    """


def extract_wikipedia_search_terms_proper_nouns(clue_text):
    """
    Extracts important entities from a crossword clue to determine what Wikipedia page to search.

    :param clue_text: The crossword clue as a string
    :return: A list of detected entities suitable for Wikipedia search
    """
    # Process the text through spaCy
    doc = nlp(clue_text)

    # Extract named entities (NER)
    named_entities = [ent.text for ent in doc.ents]

    # Extract proper nouns (e.g., "Stark" in "___ Stark")
    proper_nouns = [token.text for token in doc if token.pos_ == "PROPN"]

    # Merge named entities and proper nouns (avoid duplicates)
    search_terms = list(set(proper_nouns))

    return search_terms


def extract_wikipedia_search_terms_named_entities(clue_text):
    """
    Extracts important entities from a crossword clue to determine what Wikipedia page to search.

    :param clue: The crossword clue as a string
    :return: A list of detected entities suitable for Wikipedia search
    """
    # Process the text through spaCy
    doc = nlp(clue_text)

    # Extract named entities (NER)
    named_entities = [ent.text for ent in doc.ents]

    # Extract proper nouns (e.g., "Stark" in "___ Stark")
    proper_nouns = [token.text for token in doc if token.pos_ == "PROPN"]

    # Merge named entities and proper nouns (avoid duplicates)
    search_terms = list(set(named_entities))

    return search_terms


def extract_wikipedia_search_terms(clue_text):
    """
    Extracts important entities from a crossword clue to determine what Wikipedia page to search.

    :param clue: The crossword clue as a string
    :return: A list of detected entities suitable for Wikipedia search
    """
    # Process the text through spaCy
    doc = nlp(clue_text)

    # Extract named entities (NER)
    named_entities = [ent.text for ent in doc.ents]

    # Extract proper nouns (e.g., "Stark" in "___ Stark")
    proper_nouns = [token.text for token in doc if token.pos_ == "PROPN"]

    # Merge named entities and proper nouns (avoid duplicates)
    search_terms = list(set(named_entities + proper_nouns))

    return search_terms

def find_in_wiki(wiki, clues_df):

    for _, row in clues_df.iterrows():
        clue = row['Clue']
        answer = row['Word'].lower()

        print(f"\n-----------------------------------------------------------------------------------")
        print(f"* Clue:          {clue}")
        print(f"* Answer:        {answer}")
        print(f"* Possible List: {row['proper_nouns_and_named_entities']}")

        for phrase in row["proper_nouns_and_named_entities"]:

            page = wiki.page(phrase)
            if page is None:
                print(f"Wikipedia page not found for: {phrase}.")

            summary = page.text[:100]  # Show only first 500 chars for readability
            print(f'Wikipedia Summary for "{phrase}":\n"{summary}"\n')

            # Convert Wikipedia text to lowercase for case-insensitive search
            page_text_lower = page.text.lower()
            words_in_page = set(page_text_lower.split())  # Set of words for whole-word search
            stripped_page_text = page_text_lower.replace(" ", "")  # Remove all spaces for stripped search

            # Whole-word search
            if answer in words_in_page:
                print(f"✅ Whole-word match found for '{answer}' on Wikipedia page: {phrase}")

                # Print occurrences with 30 characters before and after
                matches = [m.start() for m in re.finditer(rf'\b{re.escape(answer)}\b', page_text_lower)]
                for match in matches:
                    start = max(0, match - 30)  # Ensure we don't go out of bounds
                    end = min(len(page.text), match + len(answer) + 30)
                    print(f"...{page.text[start:end]}...")

            # Space-stripped search
            if answer in stripped_page_text:
                print(f'✅ Space-stripped match found for "{answer}" on Wikipedia page: "{phrase}"')

                # Find occurrences of space-stripped answer in space-stripped text
                stripped_matches = [m.start() for m in re.finditer(re.escape(answer), stripped_page_text)]

                for match in stripped_matches:
                    # Convert stripped index to original text index
                    original_index = page_text_lower.find(answer, match)

                    if original_index != -1:
                        start = max(0, original_index - 30)  # Ensure we don't go out of bounds
                        end = min(len(page.text), original_index + len(answer) + 30)
                        print(f"...{page.text[start:end]}...")

            x=input("Type 1 to quit:")
            if x == "1":
                return

'''
time_init = time.time()
clues = get_clues_dataframe()
clues["Clue_clean"] = clues["Clue"].str.replace(rf"[{string.punctuation}]", "", regex=True)

# Then filter based on presence of a capital letter after the first character
clues = clues[clues["Clue_clean"].str[1:].str.contains(r'[A-Z]')].reset_index(drop=True)

# Optional: drop the cleaned column if you don't need it
clues = clues.drop(columns=["Clue_clean"])

clues = clues.head(100000)
clues["proper_nouns"] = clues["Clue"].apply(extract_wikipedia_search_terms_proper_nouns)
clues["named_entities"] = clues["Clue"].apply(extract_wikipedia_search_terms_named_entities)
#clues["proper_nouns_and_named_entities"] = clues["Clue"].apply(extract_wikipedia_search_terms)
time_end = time.time()

clues.to_excel("Wikipedia named entity search2.xlsx")
'''