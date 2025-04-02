from itertools import product
from collections import defaultdict
from letter_pattern_matching import find_words_by_pattern


# Each constraint is a tuple: (word_index1, word_index2, index_in_word1, index_in_word2)
def find_multi_intersections(patterns, constraints):
    candidate_lists = [find_words_by_pattern(p) for p in patterns]

    valid_combinations = []

    # Check all combinations of one word from each list
    for words in product(*candidate_lists):
        valid = True
        for w1_idx, w2_idx, i1, i2 in constraints:
            if len(words[w1_idx]) <= i1 or len(words[w2_idx]) <= i2:
                valid = False
                break
            if words[w1_idx][i1] != words[w2_idx][i2]:
                valid = False
                break
        if valid:
            valid_combinations.append(words)

    if not valid_combinations:
        print("No intersecting combinations found.")
    return valid_combinations

def print_as_array(results):
    result_array = [list(group) for group in results]
    print("[")
    for group in result_array:
        print(f"  {group},")
    print("]")

# base word to select which word, the answers are grouped by
def get_grouped_partner_sets(results, base_word_index=0, print_output=True):
    grouped = defaultdict(list)

    for combo in results:
        base_word = combo[base_word_index]
        partners = [w for i, w in enumerate(combo) if i != base_word_index]
        grouped[base_word].append(partners)

    if print_output:
        for base_word, partner_sets in grouped.items():
            print(f"{base_word}: [")
            for partners in partner_sets:
                print(f"  {partners},")
            print("]\n")

    return grouped

######################################################################################################
# Example Usage
######################################################################################################

patterns = [
    "..a..",   # word0 → middle letter 'a'
    "a...e",   # word1 → starts with 'a'
    ".a.e"     # word2 → second letter is 'a'
]

# (word_index1, word_index2, index_in_word1, index_in_word2)
#  The letter at position index_in_word1 of word_index1 must be equal to the letter at position index_in_word2 of word_index2.
# e.g. (0, 2, 1, 1) The letter at index 1 of word0 must equal the letter at index 1 of word2
# or word0[1] == word2[1]
constraints = [
    (0, 1, 2, 0), # word0[2] == word1[0]
    (0, 2, 1, 1)  # word0[1] == word2[1]
]

results = find_multi_intersections(patterns, constraints)

for combo in results:
    print(combo)

grouped_dict = get_grouped_partner_sets(results, base_word_index=0)


patterns = [
    "..a..",   # word0 
    "a...e",   # word1
]

constraints = [
    (0, 1, 2, 0), # word0[2] == word1[0]
]

results = find_multi_intersections(patterns, constraints)

for combo in results:
    print(combo)

grouped_dict = get_grouped_partner_sets(results, base_word_index=0)



patterns = [
    "ca.t",   # word0 
    ".a.z",   # word1 
    "t.r."    # word2 
]

constraints = [
    (0, 1, 1, 1),  # word0[1] == word1[1]
    (0, 2, 2, 2)   # word0[2] == word2[2]
]

results = find_multi_intersections(patterns, constraints)

for combo in results:
    print(combo)

grouped_dict = get_grouped_partner_sets(results, base_word_index=0)

patterns = [
    "...t",   # word0 
    ".a.z",   # word1 
    "torn"    # word2 
]

constraints = [
    (0, 1, 1, 1),  # word0[1] == word1[1]
    (0, 2, 2, 2)   # word0[2] == word2[2]
]

results = find_multi_intersections(patterns, constraints)

for combo in results:
    print(combo)

grouped_dict = get_grouped_partner_sets(results, base_word_index=0)
