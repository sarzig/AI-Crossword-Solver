from collections import defaultdict
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from tqdm import tqdm

from BERT_models.cluebert_model import ClueBertRanker
from clue_classification_and_processing.helpers import get_project_root
from clue_solving.custom_BERT_ranking import generate_variables_domains_constraints_from_crossword_ranked, get_true_answers, print_clues_and_answers
from clue_solving.csp_pattern_matching import get_filled_words, load_crossword_from_file_with_answers, solve_in_two_phases
from grid_solving.pruning_on_overlaps import get_overlapping_clues, lowercase_solutions, merge_matching_solutions, prune_on_multiple_overlaps
from puzzle_objects.crossword_and_clue import get_subset_overlap
from testing_results.auto_place_clues import auto_place_clues
from testing_results.csp_mini_testing import auto_place_non_dict_words


reg_loc = f"{get_project_root()}/data/puzzle_samples/processed_puzzle_samples/crossword_2022_07_24.csv"

cw = load_crossword_from_file_with_answers(reg_loc)
cw.enrich_clue_df()

clues = cw.clue_df

print(f"clues: {clues}")

#------ subset 2

subset1 = cw.subset_crossword("1-Across", branching_factor=1, overlap_threshold=1, return_type="crossword")


clue_answer_map = {
"1-Across": "acres",
"3-Down": "rorschachcards",
"1-Down": "aphid",
# "2-Down": "credo"
}

auto_place_clues(subset1, clue_answer_map)


subset1.detailed_print()

subset1_clues = subset1.clue_df
print(f"subset1_clues: {subset1_clues}")

filled = get_filled_words(subset1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-SST-2").to(device)
# model.eval()

from transformers import BertTokenizerFast, BertModel
# Load custom model and tokenizer
model_path = f"{get_project_root()}/BERT_models/cluebert-ranker"
tokenizer = BertTokenizerFast.from_pretrained(model_path)
# model = BertModel.from_pretrained(model_path).to(device)

model = ClueBertRanker()
model.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=device))
model.to(device)
model.eval()

variables1, domains1, constraints1 = generate_variables_domains_constraints_from_crossword_ranked(
subset1, tokenizer=tokenizer, model=model, top_k=100, device=device
)

for var, domain in domains1.items():

    if var not in filled:
        print(f"\n{var} has {len(domain)} options → {domain}...\n")


true_answers = get_true_answers(subset1)
# # Normalize true answers for comparison
true_cleaned = {k: v.strip().lower() for k, v in true_answers.items()}
missing = []

for var, true_ans in true_cleaned.items():
    domain = domains1.get(var, [])
    if true_ans.strip().lower() not in [d.strip().lower() for d in domain]:
        missing.append((var, true_ans, domain))

if missing:
    print("⚠️ True answers not found in domain after BERT ranking:")
    for var, true_ans, domain in missing:
        print(f" {var}: {true_ans} ❌ not in domain of size {len(domain)}")
else:
    print("✅ All true answers are in the CSP domains!")
    subset1_solutions = solve_in_two_phases(
    subset1,
    variables1,
    domains1,
    constraints1,
    domain_threshold=100,
    verbose=True,
    assignment=filled
    # freq_table=freq_table,
    )
    print(f"subset1 solutions: {subset1_solutions}")
    print(f"subset1 solutions len: {len(subset1_solutions)}")

# filled = get_filled_words(subset1)

# solutions = solve_in_two_phases(
# variables,
# domains,
# constraints,
# domain_threshold=100,
# verbose=True,
# # freq_table=freq_table,
# assignment=filled
# )
# print(f"subset1 solutions: {solutions}")

cw.detailed_print()


#------ subset 2
subset2 = cw.subset_crossword("2-Down", branching_factor=1, overlap_threshold=1, return_type="crossword")
subset2.detailed_print()

subset2_clues = subset2.clue_df
print(f"subset2_clues: {subset2_clues}")

# auto_place_clues(subset2, clue_answer_map)

print_clues_and_answers(subset2)

clue_answer_map2 = {

    #------- from subset 1
    "1-Across": "acres",
    "1-Down": "aphid",
    # "2-Down": "credo",

    #------ for subset 2
    "22-Across": "heroworshipper",
    # "27-Across": "ids",


    # # Test
    # "16-Across": "promo",
    # "31-Across": "dock",
    # "4-Down": "emo"


    }

auto_place_clues(subset2, clue_answer_map2)
# auto_place_non_dict_words(subset2)

filled_2 = get_filled_words(subset2)

print(f"filled_2: {filled_2}")


variables2, domains2, constraints2 = generate_variables_domains_constraints_from_crossword_ranked(
    subset2, tokenizer=tokenizer, model=model, top_k=100, device=device
)

print(f"len subset 2: {len(domains2.items())}")
print(f"domains.items(): {(domains2.items())}")

for var, domain in domains2.items():

    if var not in filled:
        print(f"\n{var} has {len(domain)} options → {domain}...\n")

print("\n🔎 Checking domain for 27-Across explicitly:")
print("27-Across" in domains2)  # Should be True
print(f"Domain for 27-Across: {domains2.get('27-Across')}")


true_answers = get_true_answers(subset2)
# # Normalize true answers for comparison
true_cleaned = {k: v.strip().lower() for k, v in true_answers.items()}
missing = []

for var, true_ans in true_cleaned.items():
    domain = domains2.get(var, [])
    if true_ans.strip().lower() not in [d.strip().lower() for d in domain]:
        missing.append((var, true_ans, domain))

if missing:
    print("⚠️ True answers not found in domain after BERT ranking:")
    for var, true_ans, domain in missing:
        print(f" {var}: {true_ans} ❌ not in domain of size {len(domain)}")
else:
    print("✅ All true answers are in the CSP domains!")
    subset2_solutions = solve_in_two_phases(
    subset2,
    variables2,
    domains2,
    constraints2,
    domain_threshold=100,
    verbose=True,
    assignment=filled_2
    # freq_table=freq_table,
    )
    print(f"subset2 solutions: {subset2_solutions}")
    print(f"subset2 solutions len: {len(subset2_solutions)}")

    print(f"subset2_clues: {subset2_clues}")
    print(f"subset2_true answers: {true_answers}")

    print(f"subset2_filled: {filled_2}")


overlap = get_overlapping_clues(subset1_solutions, subset2_solutions)
print(f"overlap: {overlap}")

# Normalize input solutions
subset1_solutions = lowercase_solutions(subset1_solutions)
subset2_solutions = lowercase_solutions(subset2_solutions)

subset1_pruned, subset2_pruned = prune_on_multiple_overlaps(
    subset1_solutions,
    subset2_solutions,
    overlap
)

final_solutions_subset1_subset2 = merge_matching_solutions(subset1_pruned, subset2_pruned, overlap)
 
print(f"final_solutions: {final_solutions_subset1_subset2}")
print(f"len final_solutions: {len(final_solutions_subset1_subset2)}")

print(f"len subset1_solutions: {len(subset1_solutions)}")
print(f"len subset2_solutions: {len(subset2_solutions)}")

subset1.detailed_print()

subset2.detailed_print()

# 1. Get and normalize true answers
true1 = {k: v.lower() for k, v in get_true_answers(subset1).items()}
true2 = {k: v.lower() for k, v in get_true_answers(subset2).items()}

# 2. Merge both true sets (subset1 and subset2)
combined_true = {**true1, **true2}

# 3. Check if it exists in final solutions
match_found = False
for i, sol in enumerate(final_solutions_subset1_subset2):
    if all(sol.get(k) == v for k, v in combined_true.items()):
        print(f"✅ Match found in final_solutions[{i}]: {sol}")
        match_found = True
        break

if not match_found:
    print("❌ Correct solution not found in final_solutions.")
    print(f"Expected solution: {combined_true}")

#################### Combine 3rd with lots of overlap ######################

# subset3 = cw.subset_crossword("22-Across", branching_factor=1, overlap_threshold=1, return_type="crossword")

# "4-Down" YES , "5-Down", "31-Across", "27-Across" 

#['1-Across', '2-Down', '16-Across', '31-Across', '27-Across', '4-Down']
# "1-Down", "16-Across",
subset3 = cw.subset_crossword("5-Down", branching_factor=1, overlap_threshold=1, return_type="crossword")

subset3.detailed_print()

subset3_clues = subset3.clue_df

# auto_place_clues(subset2, clue_answer_map)

print_clues_and_answers(subset3)

placed_clues = {**clue_answer_map, **clue_answer_map2}
print(f"placed clues:\n{placed_clues}")


auto_place_clues(subset3, placed_clues)

subset3.detailed_print()

filled_3 = get_filled_words(subset3)



variables3, domains3, constraints3 = generate_variables_domains_constraints_from_crossword_ranked(
    subset3, tokenizer=tokenizer, model=model, top_k=2000, device=device
)

print(f"len subset 3: {len(domains3.items())}")
print(f"domains.items(): {(domains3.items())}")

for var, domain in domains3.items():

    if var not in filled_3:
        print(f"\n{var} has {len(domain)} options → {domain}...\n")


true_answers = get_true_answers(subset3)
# # Normalize true answers for comparison
true_cleaned = {k: v.strip().lower() for k, v in true_answers.items()}
missing = []

for var, true_ans in true_cleaned.items():
    domain = domains3.get(var, [])
    if true_ans.strip().lower() not in [d.strip().lower() for d in domain]:
        missing.append((var, true_ans, domain))

if missing:
    print("⚠️ True answers not found in domain after BERT ranking:")
    for var, true_ans, domain in missing:
        print(f" {var}: {true_ans} ❌ not in domain of size {len(domain)}")
else:
    print("✅ All true answers are in the CSP domains!")
    subset3_solutions = solve_in_two_phases(
    subset3,
    variables3,
    domains3,
    constraints3,
    domain_threshold=100,
    verbose=True,
    assignment=filled_3
    # freq_table=freq_table,
    )
    print(f"subset3 solutions: {subset3_solutions}")
    print(f"subset3 solutions len: {len(subset3_solutions)}")

    print(f"subset3_clues: {subset3_clues}")
    print(f"subset3_true answers: {true_answers}")

    print(f"subset3_filled: {filled_3}")



overlap = get_overlapping_clues(final_solutions_subset1_subset2, subset3_solutions)
print(f"overlap: {overlap}")

# Normalize input solutions
final_solutions_subset1_subset2 = lowercase_solutions(final_solutions_subset1_subset2)
subset3_solutions = lowercase_solutions(subset3_solutions)

final_solutions_subset1_subset2_pruned, subset3_pruned = prune_on_multiple_overlaps(
    final_solutions_subset1_subset2,
    subset3_solutions,
    overlap
)

print(f"len final_solutions_subset1_subset2: {len(final_solutions_subset1_subset2)}")
print(f"len subset3_solutions: {len(subset3_solutions)}")
print(f"len final_solutions_subset1_subset2_pruned: {len(final_solutions_subset1_subset2_pruned)}")
print(f"len subset3_pruned: {len(subset3_pruned)}")

final_solutions_subset123 = merge_matching_solutions(final_solutions_subset1_subset2_pruned, subset3_pruned, overlap)

print(f"final_solutions_subset123: {final_solutions_subset123}")
print(f"len final_solutions_subset123: {len(final_solutions_subset123)}")

print(f"len final_solutions_subset1_subset2: {len(final_solutions_subset1_subset2)}")
print(f"len subset3_solutions: {len(subset3_solutions)}")
