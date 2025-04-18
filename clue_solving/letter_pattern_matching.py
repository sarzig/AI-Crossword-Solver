import re
from nltk.corpus import words

from clue_classification_and_processing.helpers import get_project_root

# Load English words
word_list = set(words.words())  

# Convert all words to lowercase
word_list = set(word.lower() for word in words.words())

# Load NYT answers from a file saved earlier 
def load_nyt_answers(path=f"{get_project_root()}/data/nyt_vocabulary.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())

nyt_answers = load_nyt_answers()  # Load answers

# Merge both vocabularies
combined_vocab = word_list.union(nyt_answers)

# with open("combined_vocab.txt", "w", encoding="utf-8") as f:
#     for word in sorted(combined_vocab):
#         f.write(word + "\n")

# print(f"Original NLTK vocab: {len(word_list):,}")
# print(f"NYT answers: {len(nyt_answers):,}")
# print(f"Combined vocab: {len(combined_vocab):,}")

# Finds words that match a given crossword pattern.
# A regex pattern (e.g., "c.t" for words like 'cat' and 'cut')
# returns a list of matching words.
def find_words_by_pattern(pattern):
    pattern = pattern.lower()
    regex = re.compile("^" + pattern.replace(".", "[a-z]") + "$")
    return [word for word in combined_vocab if regex.fullmatch(word)]


# # print("tab" in word_list)
# # print("broth" in word_list)
# # print("jazzy" in word_list)
# # print("tiara" in word_list)
# # print("blitz" in word_list)
print("egrets" in combined_vocab)

# print(len(word_list))



######################################################################################################
# Example Usage
######################################################################################################

# # Five-letter words starting with "t"
# pattern = "t...."  
# # Possible Output: ['tiger', 'table', 'twist']
# print(find_words_by_pattern(pattern))  

# pattern2 = "t..f..a"
# print(find_words_by_pattern(pattern2))  

# pattern3 = "ta."
# print(find_words_by_pattern(pattern3))  
