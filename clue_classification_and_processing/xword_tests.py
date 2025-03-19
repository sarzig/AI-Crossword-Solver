import re
import tabulate
import nltk
from nltk.corpus import wordnet
import pandas as pd
nltk.download('wordnet')
import kagglehub
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Download latest version
#path = kagglehub.dataset_download("duketemon/wordnet-synonyms")

#print("Path to dataset files:", path)


synonyms = []
for syn in wordnet.synsets("examples"):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())
#print(set(synonyms))
# Expected output: {'model', 'example', 'instance', 'illustration'}

file = r"C:\Users\switzig\Downloads\old\nytcrosswords.csv"
clues = pd.read_csv(file, encoding='latin1')

clues["ends in question"] = clues["Clue"].str[-1] == "?"
clues["number words"] = clues["Clue"].str.split().apply(len)
clues["length of clue"] = clues["Clue"].str.len()
clues["no alphabet characters"] = clues["Clue"].apply(lambda x: len(re.sub(r'[A-Za-z]', '', x)))
clues["primarily number clue"] = clues["Clue"].apply(lambda x: sum(c.isdigit() for c in x) > len(x) / 2)
clues["is quote"] = clues["Clue"].str.endswith('"') & clues["Clue"].str.startswith('"')
clues["contains underscore"] = clues["Clue"].str.contains(r"_", case=False, na=False)
clues["contains asterisk"] = clues["Clue"].str.contains(r"\*", case=False, na=False)
clues["contains done"] = clues["Clue"].str.contains(r"done", case=False, na=False)

boolean_columns = [
    "ends in question",
    "primarily number clue",
    "is quote",
    "contains underscore",
    "contains asterisk",
    "contains done"
]

# Filter the DataFrame to keep only rows where at least one condition is True
filtered_clues = clues[clues[boolean_columns].any(axis=1)]

single_word_clues = clues[
    (clues["number words"] == 1) &
    (~clues["contains underscore"]) &
    (~clues["ends in question"]) &
    (~clues["is quote"]) &
    (clues["no alphabet characters"] == 0)
]

single_word_clues = single_word_clues.loc[
    single_word_clues.astype({"Word": str, "Clue": str})
    .apply(lambda row: frozenset([row["Word"], row["Clue"]]), axis=1)
    .drop_duplicates()
    .index
]

subset = single_word_clues[["Date", "Word", "Clue"]]
#single_word_clues[["Date", "Word", "Clue"]].to_excel("single_word_clues.xlsx")
