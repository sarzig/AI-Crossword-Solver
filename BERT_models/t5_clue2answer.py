from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import pandas as pd
import torch
from transformers import TrainerCallback

from clue_classification_and_processing.helpers import get_project_root

import torch

class LivePredictionCallback(TrainerCallback):
    def __init__(self, tokenizer, dataset, model, print_every=500):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.model = model
        self.print_every = print_every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.print_every == 0 and state.global_step > 0:
            self.print_sample(state.global_step)

    def on_epoch_end(self, args, state, control, **kwargs):
        self.print_sample(f"epoch-{state.epoch:.2f}")

    def print_sample(self, step_label):
        sample = self.dataset[0]
        input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(self.model.device)
        generated_ids = self.model.generate(input_ids, max_length=16)
        clue = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        pred = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"\n--- Prediction @ {step_label} ---")
        print(f"Clue:      {clue}")
        print(f"Prediction:{pred}")
        print(f"Target:    {sample.get('target', '[Unknown]')}")
        print("-----------------------------")

class MemoryCheckCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        print(f"Step {state.global_step}: Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


print("Torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0))
print("CUDA available:", torch.cuda.is_available())
print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

from evaluate import load

metric = load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return metric.compute(predictions=decoded_preds, references=decoded_labels)

# 1. Load your crossword dataset
df = pd.read_csv(f"{get_project_root()}/data/nytcrosswords.csv", encoding="ISO-8859-1") 
df = df.dropna(subset=["Clue", "Word"])  # Drop rows missing data

# 2. Prepare data for T5 (text-to-text format)
df["input"] = df["Clue"].astype(str)
df["target"] = df["Word"].astype(str)

# 3. Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df[["input", "target"]])
dataset = dataset.train_test_split(test_size=0.05)

# 4. Load T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=64)
model = T5ForConditionalGeneration.from_pretrained("t5-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 5. Preprocessing function for tokenization
def preprocess(example):
    input_enc = tokenizer(example["input"], max_length=64, padding="max_length", truncation=True)
    target_enc = tokenizer(example["target"], max_length=16, padding="max_length", truncation=True)

    input_enc["labels"] = [
        (l if l != tokenizer.pad_token_id else -100)
        for l in target_enc["input_ids"]
    ]
    return input_enc



# 6. Tokenize the dataset
tokenized = dataset.map(preprocess, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# 7. Training arguments
training_args = TrainingArguments(
    output_dir="t5-clue2answer",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    gradient_accumulation_steps=3,
    num_train_epochs=3,
    # fp16=torch.cuda.is_available(),  # Mixed precision if GPU available
    bf16=True, # For 5090
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    push_to_hub=False,
    learning_rate=3e-5,
    load_best_model_at_end=True,
)

# 8. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
    callbacks=[MemoryCheckCallback(), 
               LivePredictionCallback(tokenizer, tokenized["test"], model, print_every=500)],
    
)

print(next(model.parameters()).device)  # Should say cuda:0 after trainer setup


# 9. Train!
trainer.train()

# 10. Save final model and tokenizer
model.save_pretrained(f"{get_project_root()}/BERT_models/t5-clue2answer-final")
tokenizer.save_pretrained(f"{get_project_root()}/BERT_models/t5-clue2answer-final")
