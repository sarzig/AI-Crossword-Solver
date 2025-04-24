from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from tqdm import tqdm

from clue_classification_and_processing.helpers import get_project_root

# 1. Load fine-tuned T5 model and tokenizer
model_dir = f"{get_project_root()}/BERT_models/t5-clue2answer-final"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
model.eval()

# 2. Test set: List of (clue, expected_answer)
test_set = [
    ("Capital of France", "paris"),
    ("Opposite of cold", "hot"),
    ("Red fruit used in pies", "apple"),
    ("Vehicle with two wheels", "bike"),
    ("Famous wizard with glasses", "harry"),
    ("Color of the sky on a clear day", "blue"),
    ("Flying mammal", "bat"),
    ("A reptile that changes color", "chameleon"),
    ("Planet we live on", "earth"),
    ("Fastest land animal", "cheetah"),
]

# 3. Run evaluation
correct = 0

print("\n📋 Test Results:\n" + "-"*50)
for clue, true_answer in tqdm(test_set):
    input_text = f"clue: {clue}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=16)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()

    result = "✅" if pred == true_answer.lower() else "❌"
    if result == "✅":
        correct += 1

    print(f"{result} Clue: '{clue}' → Prediction: '{pred}' | True: '{true_answer}'")

# 4. Accuracy
accuracy = correct / len(test_set) * 100
print(f"\n🎯 Accuracy: {correct}/{len(test_set)} correct ({accuracy:.2f}%)")


print(tokenizer("clue: Fast animal"))
print(tokenizer("cheetah"))  # Try to see if you're getting valid tokens

