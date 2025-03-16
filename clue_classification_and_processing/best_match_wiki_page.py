from clue_classification_and_processing.clue_features import get_clues_dataframe
import spacy

clues = get_clues_dataframe()

# Load spaCy English model with Named Entity Recognition (NER)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def extract_wikipedia_search_terms_proper_nouns(clue):
    """
    Extracts important entities from a crossword clue to determine what Wikipedia page to search.

    :param clue: The crossword clue as a string
    :return: A list of detected entities suitable for Wikipedia search
    """
    # Process the text through spaCy
    doc = nlp(clue)

    # Extract named entities (NER)
    named_entities = [ent.text for ent in doc.ents]

    # Extract proper nouns (e.g., "Stark" in "___ Stark")
    proper_nouns = [token.text for token in doc if token.pos_ == "PROPN"]

    # Merge named entities and proper nouns (avoid duplicates)
    search_terms = list(set(proper_nouns))

    return search_terms

def extract_wikipedia_search_terms_named_entities(clue):
    """
    Extracts important entities from a crossword clue to determine what Wikipedia page to search.

    :param clue: The crossword clue as a string
    :return: A list of detected entities suitable for Wikipedia search
    """
    # Process the text through spaCy
    doc = nlp(clue)

    # Extract named entities (NER)
    named_entities = [ent.text for ent in doc.ents]

    # Extract proper nouns (e.g., "Stark" in "___ Stark")
    proper_nouns = [token.text for token in doc if token.pos_ == "PROPN"]

    # Merge named entities and proper nouns (avoid duplicates)
    search_terms = list(set(named_entities))

    return search_terms

clues = clues.head(1000)

clues["proper_nouns"] = clues["Clue"].apply(extract_wikipedia_search_terms_proper_nouns)
clues["named_entities"] = clues["Clue"].apply(extract_wikipedia_search_terms_named_entities)

