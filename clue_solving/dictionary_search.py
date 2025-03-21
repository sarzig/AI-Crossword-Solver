
######################################################################################################
# Basic Wordnet - All words directly mentioned in clue 
######################################################################################################
import nltk
from nltk.corpus import wordnet
import torch

nltk.download("wordnet")

def find_words_by_definition(clue):
    """
    Finds words whose definitions contain all the words in the given clue.

    :param clue: The definition or crossword clue.
    :return: A list of words that match the definition.
    """
    matches = set()
    clue_words = set(clue.lower().split())  # Convert clue to a set of lowercase words

    for synset in wordnet.all_synsets():  # Iterate over all WordNet synsets (word meanings)
        definition_words = set(synset.definition().lower().split())  # Split definition into words

        # Check if ALL words in the clue appear in the definition
        if clue_words.issubset(definition_words):
            for lemma in synset.lemmas():  # Extract possible words
                matches.add(lemma.name().replace("_", " "))  # Normalize words

    return list(matches)


######################################################################################################
# Basic Wordnet - Example Usage
######################################################################################################

clue = "large feline"
matching_words = find_words_by_definition(clue)
print(f"Words matching {clue}: {matching_words}")

clue1 = "big cat"
matching_words1 = find_words_by_definition(clue1)
print(f"Words matching {clue1}: {matching_words1}")

clue2 = "spotted feline"
matching_words2 = find_words_by_definition(clue2)
print(f"Words matching {clue2}: {matching_words2}")

######################################################################################################
# BERT - All words not directly mentioned in clue 
######################################################################################################
from sentence_transformers import SentenceTransformer, util

# Load pre-trained BERT model
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

def find_words_by_bert_similarity(clue, top_n=10):
    """
    Finds words whose definitions are semantically similar to the given clue.
    
    :param clue: The definition or crossword clue.
    :param top_n: Number of closest matches to return.
    :return: A ranked list of words with their similarity scores.
    """
    matches = []
    clue_embedding = bert_model.encode(clue, convert_to_tensor=True)  # Encode the clue

    # Store WordNet definitions and words
    word_definitions = []
    word_list = []

    for synset in wordnet.all_synsets():  # Iterate over all WordNet words
        definition = synset.definition()
        if definition:  # Ignore empty definitions
            word_definitions.append(definition)
            word_list.append(synset.lemmas()[0].name().replace("_", " "))  # Get first lemma

    # Encode all definitions with BERT
    definition_embeddings = bert_model.encode(word_definitions, convert_to_tensor=True)

    # Compute cosine similarity between the clue and all definitions
    similarities = util.pytorch_cos_sim(clue_embedding, definition_embeddings)[0]

    # Sort by highest similarity
    sorted_indices = torch.topk(similarities, top_n).indices.tolist()

    for idx in sorted_indices:
        matches.append((word_list[idx], word_definitions[idx], similarities[idx].item()))

    return matches  # List of (word, definition, similarity score)


######################################################################################################
# BERT - Example Usage
######################################################################################################

# Example usage
clue = "a large feline"
matching_words = find_words_by_bert_similarity(clue, top_n=5)

# Print results
print("\nWords matching definition (BERT-based similarity):")
for word, definition, score in matching_words:
    print(f"{clue} - {word} (Score: {score:.3f}): {definition}")

# Example usage
clue1 = "big cat"
matching_words1 = find_words_by_bert_similarity(clue1, top_n=5)

# Print results
print("\nWords matching definition (BERT-based similarity):")
for word, definition, score in matching_words1:
    print(f"{clue1}- {word} (Score: {score:.3f}): {definition}")

# Example usage
clue2 = "spotted feline"
matching_words2 = find_words_by_bert_similarity(clue2, top_n=5)

# Print results
print("\nWords matching definition (BERT-based similarity):")
for word, definition, score in matching_words2:
    print(f"{clue2}- {word} (Score: {score:.3f}): {definition}")

