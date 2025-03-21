import re
from nltk.corpus import words

# Load English words
word_list = set(words.words())  

# Finds words that match a given crossword pattern.
# A regex pattern (e.g., "c.t" for words like 'cat' and 'cut')
# returns a list of matching words.
def find_words_by_pattern(pattern):

    regex = re.compile(pattern)
    return [word for word in word_list if regex.fullmatch(word)]

######################################################################################################
# Example Usage
######################################################################################################

# Five-letter words starting with "t"
pattern = "t...."  
# Possible Output: ['tiger', 'table', 'twist']
print(find_words_by_pattern(pattern))  

pattern2 = "t..f..a"
print(find_words_by_pattern(pattern2))  
