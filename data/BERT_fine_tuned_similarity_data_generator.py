import pandas as pd
import random
import nltk
from nltk.corpus import words
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch

nltk.download('words')

# Load NYT crossword data
df = pd.read_csv("data/nytcrosswords.csv", encoding="ISO-8859-1")
df = df.dropna(subset=["Clue", "Word"])
df["Clue"] = df["Clue"].str.strip()
df["Word"] = df["Word"].str.strip().str.lower()
df = df[df["Word"].str.len().between(3, 15)]

# Get usable English words from NLTK
english_words = set(w.lower() for w in words.words() if w.isalpha())
valid_lengths = df["Word"].str.len().unique()
english_words = [w for w in english_words if len(w) in valid_lengths]

true_answers = set(df["Word"].unique())
NEG_PER_POS = 3

# Use a sentence transformer for hard negatives
print("Loading SentenceTransformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Precompute embeddings for candidate negatives
candidate_words = [w for w in english_words if w not in true_answers]
print("Encoding candidate words...")
word_embeddings = model.encode(candidate_words, convert_to_tensor=True)

# Storage
data = []

print("Generating training pairs...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    clue = row["Clue"]
    true_ans = row["Word"]
    clue_embedding = model.encode(clue, convert_to_tensor=True)
    length = len(true_ans)

    # Add positive pair
    data.append({"clue": clue, "answer": true_ans, "label": 1})

    # Easy negatives
    easy_candidates = [w for w in english_words if len(w) == length and w not in true_answers and w != true_ans]
    easy_negatives = random.sample(easy_candidates, min(len(easy_candidates), NEG_PER_POS))
    for neg_ans in easy_negatives:
        data.append({"clue": clue, "answer": neg_ans, "label": 0})

    # Hard negatives (semantic similarity)
    hits = util.semantic_search(clue_embedding, word_embeddings, top_k=50)[0]
    hard_negatives = []
    for hit in hits:
        candidate = candidate_words[hit['corpus_id']]
        if candidate != true_ans and len(candidate) == length:
            hard_negatives.append(candidate)
        if len(hard_negatives) == NEG_PER_POS:
            break
    for neg_ans in hard_negatives:
        data.append({"clue": clue, "answer": neg_ans, "label": -1})

# Save to CSV
out_df = pd.DataFrame(data)
out_df.to_csv("bert_training_pairs_with_easy_and_hard_negatives.csv", index=False)
print("✅ Done. Saved to 'bert_training_pairs_with_easy_and_hard_negatives.csv'")
