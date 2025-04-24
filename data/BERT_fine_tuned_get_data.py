import pandas as pd
import random
import nltk
from nltk.corpus import words
from tqdm import tqdm

nltk.download('words')

# Load crossword data
df = pd.read_csv("data/nytcrosswords.csv", encoding="ISO-8859-1")
df = df.dropna(subset=["Clue", "Word"])
df["Clue"] = df["Clue"].str.strip()
df["Word"] = df["Word"].str.strip().str.lower()

# Clean: Remove overly short/long answers
df = df[df["Word"].str.len().between(3, 15)]

# Get word list from NLTK and filter for usable words
english_words = set(w.lower() for w in words.words() if w.isalpha())
valid_lengths = df["Word"].str.len().unique()
english_words = [w for w in english_words if len(w) in valid_lengths]

# Build answer lookup to avoid selecting true answers as negatives
true_answers = set(df["Word"].unique())

# Parameters
NEG_PER_POS = 3  # Number of negatives to generate per clue

# Storage
data = []

print("Generating training pairs...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    clue = row["Clue"]
    true_ans = row["Word"]
    length = len(true_ans)

    # Positive pair
    data.append({"clue": clue, "answer": true_ans, "label": 1})

    # Negative pairs
    candidates = [w for w in english_words if len(w) == length and w not in true_answers and w != true_ans]
    if len(candidates) >= NEG_PER_POS:
        negatives = random.sample(candidates, NEG_PER_POS)
    else:
        negatives = candidates  # fall back if not enough

    for neg_ans in negatives:
        data.append({"clue": clue, "answer": neg_ans, "label": 0})

# Save to CSV
out_df = pd.DataFrame(data)
out_df.to_csv("bert_training_pairs.csv", index=False)
print("✅ Done. Saved to 'bert_training_pairs.csv'")
