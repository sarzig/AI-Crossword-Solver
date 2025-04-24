import torch
from transformers import BertTokenizerFast, BertModel
from tqdm import tqdm
import pandas as pd

from clue_classification_and_processing.helpers import get_project_root
from BERT_models.BERT_similarity_ranking import ClueBertRanker


# Load model and tokenizer
# model_path = f"{get_project_root()}/BERT_models/cluebert-ranker"
model_path = "sdeakin/cluebert-ranker"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path).to(device)
score_head = torch.nn.Linear(model.config.hidden_size, 1).to(device)
# score_head.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=device), strict=False)
model.eval()

# Ranking function
def rank_answers(clue: str, candidates: list[str], top_k=10):
    inputs = [f"{clue} [SEP] {ans}" for ans in candidates]
    encodings = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt", max_length=64).to(device)

    with torch.no_grad():
        outputs = model(**encodings)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        scores = score_head(cls_embeddings).squeeze(-1)

    sorted_indices = torch.argsort(scores, descending=True)
    ranked = [(candidates[i], float(scores[i])) for i in sorted_indices[:top_k]]

    return ranked

# Example usage
if __name__ == "__main__":
    # Example clue and candidate answers
    clue = "podcasting equiptment"
    candidates = ["mic", "germany", "tokyo", "speaking", "talk", "apple", "omega", "dog", "table", "record"]

    print(f"\nClue: {clue}")
    print("Candidates:", candidates)

    ranked = rank_answers(clue, candidates)
    print("\nRanked answers:")
    for rank, (ans, score) in enumerate(ranked, 1):
        print(f"{rank}. {ans} (Score: {score:.4f})")
