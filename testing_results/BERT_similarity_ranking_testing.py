from collections import defaultdict
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from tqdm import tqdm

from clue_classification_and_processing.helpers import get_project_root
from clue_solving.BERT_similarity_ranking import generate_variables_domains_constraints_from_crossword_ranked, get_true_answers, print_clues_and_answers, rank_csp_solutions
from clue_solving.csp_pattern_matching import get_filled_words, load_crossword_from_file_with_answers, solve_in_two_phases
from clue_solving.letter_pattern_matching import find_words_by_pattern

from testing_results.auto_place_clues import auto_place_clues
from testing_results.csp_mini_testing import auto_place_non_dict_words



print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model once, move model to device
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2").to(device)

from transformers import RobertaTokenizer, RobertaForSequenceClassification

from clue_classification_and_processing.helpers import get_project_root
from clue_solving.csp_pattern_matching import load_crossword_from_file_with_answers
from testing_results.auto_place_clues import auto_place_clues

# tokenizer = RobertaTokenizer.from_pretrained("ynie/roberta-base-snli_mnli_fever_anli_R1_R2_R3-nli")
# model = RobertaForSequenceClassification.from_pretrained("ynie/roberta-base-snli_mnli_fever_anli_R1_R2_R3-nli").to(device)



model.eval()

mini_loc = f"{get_project_root()}/data/puzzle_samples/processed_puzzle_samples/mini_2024_03_02.csv"

cw = load_crossword_from_file_with_answers(mini_loc)

# Automatically detect and place non-dictionary words
auto_place_non_dict_words(cw)

# cw.place_word()
print_clues_and_answers(cw)
# 📋 Clues and Answers:
#   1-Across: Algebra, calculus, etc. → math
#   5-Across: College reunion attendees → alumni
#   8-Across: Broadway's "The Book of ____" → mormon
#   9-Across: Author Patchett → ann
#   10-Across: Nickname for a longtime Supreme Court justice → rbg
#   12-Across: Heat on low → simmer
#   14-Across: Drug such as morphine or codeine → opiate
#   15-Across: Feature of a leopard or lobster → claw
#   1-Down: Word belted out by Freddie Mercury in the first verse of "Bohemian Rhapsody" → mama
#   2-Down: Pete ___, All-Star slugger for the New York Mets → alonso
#   3-Down: Rounded root vegetable → turnip
#   4-Down: "Let me think ..." → hmm
#   6-Down: Standard → normal
#   7-Down: Still being tested, as an app → inbeta
#   11-Down: Got bigger → grew
#   13-Down: Piece of podcasting equipment → mic

clue_answer_map = {
    "1-Across": "math",
    # "5-Across": "alumni",
    # "8-Across": "mormon",
    # "9-Across": "ann",
    "10-Across": "rbg",
    "12-Across": "simmer",
    "14-Across": "opiate",
    "15-Across": "claw",
    "1-Down": "mama",
    # "2-Down": "alonso",
    # "3-Down": "turnip",
    # "4-Down": "hmm",
    # "6-Down": "normal",
    # "7-Down": "inbeta",
    # "11-Down": "grew",
    # "13-Down": "mic"
}

auto_place_clues(cw, clue_answer_map)

# Optional: visually verify
cw.detailed_print()

filled = get_filled_words(cw)
print(f"filled words: {filled}")

print("🟢 ABOUT TO CALL DOMAIN GENERATION...")
variables, domains, constraints = variables, domains, constraints = generate_variables_domains_constraints_from_crossword_ranked(
    cw, tokenizer=tokenizer, model=model, top_k=1000, device=device
)
print("✅ DOMAIN GENERATION DONE.")

for var, domain in domains.items():

    if var not in filled:
        print(f"\n{var} has {len(domain)} options → {domain}...\n")

true_answers = get_true_answers(cw)
# # Normalize true answers for comparison
true_cleaned = {k: v.strip().lower() for k, v in true_answers.items()}
missing = []

for var, true_ans in true_cleaned.items():
    domain = domains.get(var, [])
    if true_ans.strip().lower() not in [d.strip().lower() for d in domain]:
        missing.append((var, true_ans, domain))

if missing:
    print("⚠️ True answers not found in domain after BERT ranking:")
    for var, true_ans, domain in missing:
        print(f"  {var}: {true_ans} ❌ not in domain of size {len(domain)}")
else:
    print("✅ All true answers are in the CSP domains!")
    solutions = solve_in_two_phases(
    variables,
    domains,
    constraints,
    domain_threshold=100,
    verbose=True,
    # freq_table=freq_table,
    assignment=filled 
)

    if solutions:
        print(f"🎯 {len(solutions)} solution(s) returned. Reranking to find the best one...")

        # Use BERT-based scoring to pick the best overall fit
        ranked = rank_csp_solutions(cw, solutions, tokenizer, model, device)

        print(f"ranked: {ranked}")
        solution, total_score = ranked[0]

        print(f"🏆 Best CSP solution selected with total relevance score: {total_score:.4f}")

        for var, word in solution.items():
            try:
                cw.place_word(word, var)
            except Exception as e:
                print(f"⚠️ Could not place {var}: {word} → {e}")

    cw.detailed_print()