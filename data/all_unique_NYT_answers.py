from pathlib import Path
import pandas as pd

def collect_all_nyt_answers(file_path, save_to=None):
    df = pd.read_csv(file_path, encoding="ISO-8859-1")

    # Clean and standardize the answers
    if "Word" not in df.columns:
        raise ValueError("Expected 'Word' column not found in dataset")

    answers = df["Word"].dropna().astype(str).str.lower().str.strip()

    # Optional: filter out blanks or single-letter noise
    answers = answers[answers.str.len() > 1]

    # Convert to set for uniqueness
    unique_answers = set(answers)

    if save_to:
        with open(save_to, "w", encoding="utf-8") as f:
            for word in sorted(unique_answers):
                f.write(f"{word}\n")

    return unique_answers

vocab_path = "data/nyt_vocabulary.txt"
nyt_file_path = "data/nytcrosswords.csv"

nyt_vocab = collect_all_nyt_answers(nyt_file_path, save_to=vocab_path)
print(f"Collected {len(nyt_vocab)} unique answers and saved to {vocab_path}")

