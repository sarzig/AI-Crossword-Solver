from collections import defaultdict
import pandas as pd
import torch
from tqdm import tqdm
import wordninja
from transformers import BertTokenizerFast
from clue_classification_and_processing.helpers import get_project_root
from clue_solving.csp_pattern_matching import get_filled_words, load_crossword_from_file_with_answers, solve_in_two_phases
from clue_solving.letter_pattern_matching import find_words_by_pattern
from testing_results.auto_place_clues import auto_place_clues
from testing_results.csp_mini_testing import auto_place_non_dict_words
from BERT_models.cluebert_model import ClueBertRanker

def is_tokenizable(tokenizer, word):
    tokens = tokenizer.tokenize(word)
    return all(t != "[UNK]" for t in tokens)

def try_split(word):
    split = wordninja.split(word)
    if len(split) > 1 and all(len(w) > 1 for w in split):
        return " ".join(split)
    return word

def rank_candidates(clue, candidates, tokenizer, model, top_k=None, device="cuda"):
    model.eval()
    processed_inputs, original_answers = [], []

    for ans in candidates:
        processed_ans = ans if is_tokenizable(tokenizer, ans) else try_split(ans)
        processed_inputs.append(f"{clue} [SEP] {processed_ans}")
        original_answers.append(ans)

    batch = tokenizer(processed_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.inference_mode():
        outputs = model(**batch)
        scores = outputs.squeeze().detach().cpu().tolist()

    answer_scores = list(zip(original_answers, scores))
    ranked = sorted(answer_scores, key=lambda x: x[1], reverse=True)
    return [ans for ans, _ in ranked[:top_k]] if top_k else [ans for ans, _ in ranked]

def rank_csp_solutions(cw, solutions, tokenizer, model, device):
    clue_texts = {row["number_direction"]: row["clue"] for _, row in cw.clue_df.iterrows()}
    ranked = []

    for solution in tqdm(solutions, desc="📈 Ranking CSP solutions"):
        total_score = 0
        for var, word in solution.items():
            clue = clue_texts.get(var, "")
            inputs = tokenizer(f"{clue} [SEP] {word}", return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                score = outputs.squeeze().item()
            total_score += score
        ranked.append((solution, total_score))

    return sorted(ranked, key=lambda x: x[1], reverse=True)

def rank_candidates_batch(clue_answer_pairs, tokenizer, model, device="cuda", batch_size=256):
    model.eval()
    scores = []

    for i in tqdm(range(0, len(clue_answer_pairs), batch_size), desc="🔎 BERT scoring batches"):
        batch_pairs = clue_answer_pairs[i:i + batch_size]
        texts = [f"{clue} [SEP] {answer}" for clue, answer in batch_pairs]

        try:
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
                batch_scores = outputs.squeeze().detach().cpu().tolist()
                # batch_scores = outputs.logits.squeeze().detach().cpu().tolist()
            scores.extend(batch_scores)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("CUDA OOM. Try reducing batch size.")
                torch.cuda.empty_cache()
                raise e
            else:
                raise e
    return scores

def generate_variables_domains_constraints_from_crossword_ranked(cw, tokenizer, model, top_k=5, device="cuda", force_answer=False):
    clues = cw.clue_df.to_dict(orient="records")
    grid_map = defaultdict(list)
    variables, raw_domains, final_domains = [], {}, {}
    constraints = set()
    var_to_positions, var_to_clue = {}, {}
    filled = get_filled_words(cw)
    grid = cw.grid
    filled = get_filled_words(cw)

    print("Generating variables and initial domains (constraint-aware)...")
    for clue in tqdm(clues, desc="Processing clues"):
        var = clue["number_direction"]
        clue_text = clue["clue"]
        start_row, start_col = clue["start_row"], clue["start_col"]
        end_row, end_col = clue["end_row"], clue["end_col"]
        length = clue["length"]

        positions = [(start_row + i, start_col) if start_col == end_col else (start_row, start_col + i) for i in range(length)]

        variables.append(var)
        var_to_positions[var] = positions
        var_to_clue[var] = clue_text

        for i, (r, c) in enumerate(positions):
            grid_map[(r, c)].append((var, i))

        if var in filled:
            final_domains[var] = [filled[var]]
        else:
            pattern = "".join(
                grid[r][c].strip().lower() if isinstance(grid[r][c], str) and grid[r][c].strip().isalpha() else "."
                for r, c in positions
            )
            raw_domains[var] = [w.lower() for w in find_words_by_pattern(pattern)]
            print(f"🔍 {var} → pattern '{pattern}' → {len(raw_domains[var])} matches")

    print("Building constraints...")
    for square, entries in grid_map.items():
        if len(entries) > 1:
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    var1, idx1 = entries[i]
                    var2, idx2 = entries[j]
                    constraints.add((var1, var2, idx1, idx2))
                    constraints.add((var2, var1, idx2, idx1))

    print("Filtering domains using constraints...")
    for var in tqdm(variables, desc="⛓️ Filtering with overlaps"):
        if var in filled:
            continue

        positions = var_to_positions[var]
        filtered = []
        for word in raw_domains.get(var, []):
            valid = True
            for i, (r, c) in enumerate(positions):
                cell = grid[r][c]
                if isinstance(cell, str) and cell.strip() and cell != "[ ]":
                    grid_letter = cell.strip().lower()
                    if len(grid_letter) == 3 and grid_letter.startswith("[") and grid_letter.endswith("]"):
                        grid_letter = grid_letter[1].lower()  # extract letter from e.g. [F]
                    if word[i].lower() != grid_letter:
                        valid = False
                        break
            if valid:
                filtered.append(word)

        final_domains[var] = filtered

    print("BERT ranking within filtered domains...")
    clue_answer_pairs = []
    var_to_candidates = defaultdict(list)
    for var in variables:
        if var in filled:
            continue
        clue = var_to_clue[var]
        for word in final_domains[var]:
            word = word.lower()
            clue_answer_pairs.append((clue, word))
            var_to_candidates[var].append(word)

    scores = rank_candidates_batch(clue_answer_pairs, tokenizer, model, device=device)

    print("Ranking and selecting top_k...")
    i = 0
    for var in variables:
        if var in filled:
            continue
        scored = [(word, scores[i + j]) for j, word in enumerate(var_to_candidates[var])]
        i += len(var_to_candidates[var])
        final_domains[var] = [word.lower() for word, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]]

    print(f"✅ Done! Variables: {len(variables)} | Constraints: {len(constraints)}")

    true_answers = get_true_answers(cw)
    for var in variables:
        if var in filled:
            continue
        true_answer = true_answers.get(var)
        true_answer = true_answer.strip().lower() if true_answer else None

        # Safety: even if not in candidates, reinsert if forced
        domain_words = [w.strip().lower() for w in final_domains.get(var, [])]
        candidate_words = [w.strip().lower() for w in var_to_candidates.get(var, [])]

        if true_answer and true_answer not in domain_words:
            if true_answer not in candidate_words:
                print(f"⚠️ True answer '{true_answer}' for {var} not matched by pattern — not in candidates.")
            if force_answer:
                print(f"🛡️ Forcing reinsertion of true answer '{true_answer}' into domain for {var}")
                final_domains[var].append(true_answer)

        if not raw_domains.get(var) or true_answer not in [w.lower() for w in raw_domains[var]]:
            print(f"WARNING: True answer '{true_answer}' was not returned by pattern matcher for {var}")

    return variables, final_domains, list(constraints)

def get_true_answers(cw):
    answer_col = next((col for col in cw.clue_df.columns if "answer" in col.lower()), None)
    if not answer_col:
        raise ValueError("No answer column found in crossword.")
    return {
        row["number_direction"]: str(row[answer_col]).strip().lower()
        for _, row in cw.clue_df.iterrows()
        if pd.notna(row[answer_col])
    }

def compare_with_true_answers(solution, cw):
    true_answers = get_true_answers(cw)
    correct, incorrect = [], []
    for var, predicted in solution.items():
        predicted_clean = predicted.strip().lower()
        true = true_answers.get(var)
        if true is None:
            print(f"No true answer found for {var}")
            continue
        if predicted_clean == true.strip().lower():
            correct.append((var, predicted))
        else:
            incorrect.append((var, predicted, true))
    return correct, incorrect

def print_clues_and_answers(cw):
    df = cw.clue_df
    clue_col = next((col for col in df.columns if "clue" in col.lower()), None)
    answer_col = next((col for col in df.columns if "answer" in col.lower()), None)
    if not clue_col or not answer_col:
        print("❌ Could not find clue or answer column.")
        return
    print("📋 Clues and Answers:")
    for _, row in df.iterrows():
        var = row.get("number_direction", "??")
        clue = row.get(clue_col, "").strip()
        answer = row.get(answer_col, "").strip().lower()
        print(f"  {var}: {clue} → {answer}")
