import re
from nltk.corpus import words

# Load English words
word_list = set(words.words())  

# Convert all words to lowercase
word_list = set(word.lower() for word in words.words())

# Finds words that match a given crossword pattern.
# A regex pattern (e.g., "c.t" for words like 'cat' and 'cut')
# returns a list of matching words.
def find_words_by_pattern(pattern):
    pattern = pattern.lower()
    regex = re.compile("^" + pattern.replace(".", "[a-z]") + "$")
    return [word for word in word_list if regex.fullmatch(word)]


# print("tab" in word_list)
# print("broth" in word_list)
# print("jazzy" in word_list)
# print("tiara" in word_list)
# print("blitz" in word_list)
# print("shy" in word_list)

print(len(word_list))



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
