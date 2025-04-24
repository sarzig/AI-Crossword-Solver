import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import os

from BERT_models.cluebert_model import ClueBertRanker
from clue_classification_and_processing.helpers import get_project_root

# Load and prepare dataset
df = pd.read_csv(f"{get_project_root()}/BERT_models/bert_training_pairs_with_easy_and_hard_negatives.csv")

df = df.sample(n=500000, random_state=42)

df["text"] = df["clue"].astype(str) + " [SEP] " + df["answer"].astype(str)

# Group by clue to create (pos, neg) sets
grouped = df.groupby("clue")
all_clues = list(grouped.groups.keys())

# Split clues to avoid clue leakage
train_clues, val_clues = train_test_split(all_clues, test_size=0.1, random_state=42)
train_df = df[df["clue"].isin(train_clues)].reset_index(drop=True)
val_df = df[df["clue"].isin(val_clues)].reset_index(drop=True)


class ClueAnswerRankingDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=64):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.groups = dataframe.groupby("clue")

        # Only keep clues that have at least one positive and one negative (label 0 or -1)
        self.valid_clues = [
            k for k, g in self.groups
            if (g["label"] == 1).any() and (g["label"] != 1).any()
        ]

    def __len__(self):
        return len(self.valid_clues)

    def __getitem__(self, idx):
        clue = self.valid_clues[idx]
        group = self.groups.get_group(clue)

        pos_row = group[group["label"] == 1].sample(n=1).iloc[0]
        neg_row = group[group["label"] != 1].sample(n=1).iloc[0]  # hard or easy neg

        pos_input = self.tokenizer(pos_row["text"], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        neg_input = self.tokenizer(neg_row["text"], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")

        return {
            "pos_input_ids": pos_input["input_ids"].squeeze(0),
            "pos_attention_mask": pos_input["attention_mask"].squeeze(0),
            "neg_input_ids": neg_input["input_ids"].squeeze(0),
            "neg_attention_mask": neg_input["attention_mask"].squeeze(0),
            "clue": clue
        }


def evaluate(model, dataloader, tokenizer, device, top_k=3):
    model.eval()
    correct_1 = 0
    correct_k = 0
    total = 0

    for batch in tqdm(dataloader, desc="🔎 Validating"):
        with torch.no_grad():
            clues = batch["clue"]  # now a list of 32 clues

            for clue in clues:
                group = val_df[val_df["clue"] == clue]

                if len(group) < top_k:
                    continue  # skip if too few answers

                encoded = tokenizer(group["text"].tolist(), truncation=True, padding="max_length", max_length=64, return_tensors="pt").to(device)
                scores = model(encoded["input_ids"], encoded["attention_mask"])
                true_index = group["label"].values.argmax()
                ranked = torch.argsort(scores, descending=True)

                total += 1
                if ranked[0] == true_index:
                    correct_1 += 1
                if true_index in ranked[:top_k]:
                    correct_k += 1

    return {
        "top_1_acc": correct_1 / total if total > 0 else 0,
        f"top_{top_k}_acc": correct_k / total if total > 0 else 0
    }



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = ClueBertRanker().to(device)

    train_set = ClueAnswerRankingDataset(train_df, tokenizer)
    val_set = ClueAnswerRankingDataset(val_df, tokenizer)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.MarginRankingLoss(margin=1.0)

    EPOCHS = 5

    for epoch in range(EPOCHS):
        print(f"\n🌱 Epoch {epoch+1}/{EPOCHS}")
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            pos_input_ids = batch["pos_input_ids"].to(device)
            pos_mask = batch["pos_attention_mask"].to(device)
            neg_input_ids = batch["neg_input_ids"].to(device)
            neg_mask = batch["neg_attention_mask"].to(device)

            pos_scores = model(pos_input_ids, pos_mask)
            neg_scores = model(neg_input_ids, neg_mask)

            target = torch.ones(pos_scores.size()).to(device)
            loss = criterion(pos_scores, neg_scores, target)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"📉 Training Loss: {avg_loss:.4f}")

        # Validation ranking
        val_acc = evaluate(model, val_loader, tokenizer, device)
        print(f"✅ Val top-1 acc: {val_acc['top_1_acc']:.4f} | top-{3} acc: {val_acc['top_3_acc']:.4f}")

    # Save
    model_path = "BERT_models/cluebert-ranker"
    # model.bert.save_pretrained(model_path)
    torch.save(model.state_dict(), f"{model_path}/model.pt")
    tokenizer.save_pretrained(model_path)
    print(f"\n💾 Model saved to {model_path}")


if __name__ == "__main__":
    main()
