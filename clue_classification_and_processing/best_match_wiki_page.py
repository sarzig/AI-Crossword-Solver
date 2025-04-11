import os
import re
import string
import time
import openpyxl
import pandas as pd
import wikipediaapi
import spacy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from clue_classification_and_processing.helpers import get_project_root

"""
Author: Sarah

xxx tbd: can I get this working?

Purpose of this page is to find the best match Wikipedia page 
given a clue. Typically it would be a proper noun, but could also be a noun in general. 

This page also provides processing functions to pre-process the wikipedia dump page
which contains all English language titles.

* get_filename_wikipedia_dump() - getter for wikipedia dump page (english lang. titles)
* preprocess_wikipedia_page_name(line, remove_underscore=True, down_case=True, human_name_optimize=False)
  - helper for get_all_wikipedia_pages(remove_underscore=True, down_case=True, human_name_optimize=False)
* get_all_wikipedia_pages(remove_underscore=True, down_case=True, human_name_optimize=False)
  - converts the wiki title dump page (titles of al english articles)
"""

LOAD_SPACY = False
if LOAD_SPACY:

    # Load spaCy English model with Named Entity Recognition (NER)
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download

        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")


def get_filename_wikipedia_dump():
    """
    Simple helper to return link to wiki data dump (All English language article titles).
    :return: filename
    """

    filename = f"{get_project_root()}/data/wiki/enwiki-latest-all-titles-in-ns0"
    if not os.path.exists(filename):
        print(f"Wikipedia titles file not found at:\n    {filename}")
        print("download it from:\nhttps://dumps.wikimedia.org/enwiki/latest/")
        print("Filename: enwiki-latest-all-titles-in-ns0.gz")
        print(f"Save to {get_project_root()}/data/wiki/")
    return filename


def preprocess_wikipedia_page_name(title, remove_underscore=True, down_case=True, human_name_optimize=False, remove_punctuation=False):
    """
    Helper to call during get_all_wikipedia_pages.

    :param line: wikipedia page name
    :param remove_underscore: removes ALL underscores in the name (at the peril of removing
                              underscores that are semantically important)
    :param down_case: lowercase the text
    :param human_name_optimize: flips text after comma, best for human names
    :return: processed page name
    """

    # If human_name_optimize=True, and there is one comma in the line,
    # remove underscore and flip Aniston,_Jennifer -> Jennifer Aniston
    not_bounded_by_apostrophes = title[0] == '"' and title[:1] == '"'
    optimize_for_names = human_name_optimize and ",_" in title and title.count(
        ",") == 1 and not_bounded_by_apostrophes
    if optimize_for_names:
        try:
            last, first = title.split(",_")
            title = f"{first} {last}"
        except ValueError:
            print(f"value error: {title}")

    # If remove_underscore=True, then remove all underscores in a word and
    # replace with spaces
    if remove_underscore:
        title = title.replace("_", " ")

    # remove punctuation
    if remove_punctuation:
        title = title.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
        title = title.replace("  ", " ")

    # If down_case, then do that
    if down_case:
        title = title.lower()

    return title


def is_title_of_interest(title):
    """
    Returns true if length of title is greater than 1 and there is at least
    one alphanumeric character.

    :param title: title to check
    :return: True or False
    """

    pattern = re.compile(r"[a-zA-Z0-9]")  # pattern to check for alpha-numeric
    one_alphanumeric_exists = bool(pattern.search(title))
    return one_alphanumeric_exists and len(title) > 1


def get_all_wikipedia_pages_of_interest():
    """
    This looks for the wikipedia dump page and returns all pages that:
    * are at least 2 characters in length (xxx tbd)
    * contain at least 1 az/AZ character and at least 1 0-9 character.

    This will exclude pages that have emojis or foreign language text.

    :return: list of pages of interest
    """
    with (open(get_filename_wikipedia_dump(), "r", encoding="utf-8") as file):
        titles = []
        for line in file:
            stripped = line.strip()
            if is_title_of_interest(stripped):
                titles.append(stripped)

    return titles


def get_all_wikipedia_pages(input_list=None, remove_underscore=True, down_case=True, human_name_optimize=False):
    """
    Returns list of all ~20million wikipedia pages. Original file is from the wiki dump
    page https://dumps.wikimedia.org/enwiki/latest/

    Notes on flip_comma:
    English location convention is El Paso, Texas, where El Paso is a subset of Texas.
    Wikipedia refers to this as: El_Paso,_Texas

    This is programmatically indistinguishable from the human naming convention wikipedia
    uses: Aniston,_Jennifer (that is, without more advanced semantic analysis).

    In clues, El Paso would be referred to frequently as El Paso, Texas, however a person
    would NEVER be referred to as Aniston, Jennifer in a clue.

    There is therefore value in pre-processing this wikipedia page list:
    * leave commas so that place names are findable: El_Paso,_Texas -> el paso, texas
    * remove and flip commas so that human names are findable: Jennifer Aniston

    The comma flip list will be optimized for humans, and will render other
    page names poorly, for example the following clues lose some value:
    !Bienvenido,_Mr._Marshall! --> Mr. Marshall! !Bienvenido
    (It's_A)_Long,_Lonely_Highway --> Lonely Highway, (It's A) Long

    Therefore, if an entity starts with a comma, then the human_name_optimize=False
    list is preferable.

    :param line: wikipedia page name
    :param remove_underscore: removes ALL underscores in the name (at the peril of removing
                              underscores that are semantically important)
    :param down_case: lowercase the text
    :param human_name_optimize: flips text after comma, best for human names
    :return: list of all wiki pages processed per arguments
    """

    # If input_list is None, get it
    if input_list is None:
        input_list = get_all_wikipedia_pages_of_interest()

    # Iterate and process each title
    titles = []
    for title in input_list:
        titles.append(preprocess_wikipedia_page_name(title=title,
                                                     remove_underscore=remove_underscore,
                                                     down_case=down_case,
                                                     human_name_optimize=human_name_optimize))

    return titles


def make_wikipedia_lookup_dataframe():
    """
    Speeds up interaction with enwiki-latest-all-titles-in-ns0 by pre-processing into tokens,
    lowercasing, removing underscores and flipping commas.
    :return:
    """
    print("Making lookup dataframe from enwiki-latest-all-titles-in-ns0. May take several minutes...")
    original_wikipedia_title_list = get_all_wikipedia_pages_of_interest()
    no_underscore_downcase = get_all_wikipedia_pages(input_list=original_wikipedia_title_list,
                                                     remove_underscore=True,
                                                     down_case=True)
    no_underscore_downcase_human_name_optimize = get_all_wikipedia_pages(input_list=original_wikipedia_title_list,
                                                                         remove_underscore=True,
                                                                         down_case=True,
                                                                         human_name_optimize=False)

    titles_df = pd.DataFrame({
        "original": original_wikipedia_title_list,
        "no_underscore_downcase": no_underscore_downcase,
        "no_underscore_downcase_tokens": [title.split() for title in no_underscore_downcase],
        "human_name_optimize": no_underscore_downcase_human_name_optimize,
        "human_name_optimize_tokens": [title.split() for title in no_underscore_downcase_human_name_optimize],

    })

    print("make_wikipedia_lookup_dataframe completed.")

    return titles_df


def score_match(input_text, candidate_text):
    input_tokens = input_text.lower().split()
    candidate_tokens = candidate_text.lower().split()

    if input_tokens == candidate_tokens:
        return 1.0  # exact match

    # Count matching words in order
    match_count = sum(1 for i, word in enumerate(input_tokens)
                      if i < len(candidate_tokens) and word == candidate_tokens[i])

    total_words = max(len(input_tokens), len(candidate_tokens))
    if match_count == 0:
        return 0.0
    else:
        diff = total_words - match_count
        return 1 - (diff / total_words)


def find_best_wikipedia_page_lookup(text, lookup_table, top_k=500):
    """
    Fast version using pre-tokenized columns and direct loop.
    Assumes lookup_table includes *_tokens columns.
    """
    text_tokens = text.lower().split()
    len_text = len(text_tokens)

    def score(candidate_tokens):
        if text_tokens == candidate_tokens:
            return 1.0
        match_count = sum(
            1 for i, token in enumerate(text_tokens)
            if i < len(candidate_tokens) and token == candidate_tokens[i]
        )
        if match_count == 0:
            return 0.0
        total = max(len(candidate_tokens), len_text)
        return 1 - (total - match_count) / total

    scores = []
    for original, tokens1, tokens2 in zip(
        lookup_table["original"],
        lookup_table["no_underscore_downcase_tokens"],
        lookup_table["human_name_optimize_tokens"]
    ):
        score1 = score(tokens1)
        score2 = score(tokens2)
        final_score = max(score1, score2)
        if final_score > 0:
            scores.append((original, final_score))

    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores[:top_k]


def best_wikipedia_pages_for_entity_fuzzy(entity, wiki_titles, limit=5, threshold=80):
    """
    Given an entity and a list of Wikipedia page titles, returns a ranked list
    of titles that most closely match the entity using fuzzy string matching.

    # genai attempt, not sure about this one.

    :param entity: str, entity or clue text
    :param wiki_titles: list of str, all Wikipedia page titles
    :param limit: int, number of top results to return
    :param threshold: int, minimum fuzzy score to include a result
    :return: list of matching Wikipedia page titles
    """
    matches = process.extract(entity, wiki_titles, scorer=fuzz.token_sort_ratio, limit=limit)
    filtered_matches = [title for title, score in matches if score >= threshold]
    return filtered_matches if filtered_matches else None


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

            x = input("Type 1 to quit:")
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

lookup_df = make_wikipedia_lookup_dataframe()

original_wikipedia_title_list = get_all_wikipedia_pages_of_interest()
tokens_list = []

for each in original_wikipedia_title_list:
    tokens_list.append(preprocess_wikipedia_page_name(each, remove_punctuation=True).strip().split(" "))


from collections import defaultdict

all_tokens_dict = defaultdict(list)

for i in range(len(original_wikipedia_title_list)):
    for token in tokens_list[i]:
        all_tokens_dict[token].append(original_wikipedia_title_list[i])

import pickle

# Save
with open('all_tokens_dict.pkl', 'wb') as f:
    pickle.dump(all_tokens_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

# Load
with open('all_tokens_dict.pkl', 'rb') as f:
    all_tokens_dict_res = pickle.load(f)

for word in ["jennifer", "aniston"]:
    print(all_tokens_dict_res[word])