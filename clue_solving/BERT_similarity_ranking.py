from collections import defaultdict
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from tqdm import tqdm

from clue_classification_and_processing.helpers import get_project_root
from clue_solving.csp_pattern_matching import get_filled_words, load_crossword_from_file_with_answers, solve_in_two_phases
from clue_solving.letter_pattern_matching import find_words_by_pattern

from testing_results.auto_place_clues import auto_place_clues
from testing_results.csp_mini_testing import auto_place_non_dict_words


import wordninja

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

    processed_inputs = []
    original_answers = []

    for ans in candidates:
        processed_ans = ans if is_tokenizable(tokenizer, ans) else try_split(ans)
        processed_inputs.append(f"{clue} [SEP] {processed_ans}")
        original_answers.append(ans)

    batch = tokenizer(processed_inputs, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.inference_mode():
        outputs = model(**batch)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1]

    answer_scores = list(zip(original_answers, probs.cpu().numpy()))
    ranked = sorted(answer_scores, key=lambda x: x[1], reverse=True)
    return [ans for ans, _ in ranked[:top_k]] if top_k else [ans for ans, _ in ranked]


def rank_csp_solutions(cw, solutions, tokenizer, model, device):
    """
    Given a Crossword object and multiple full CSP solutions,
    return them ranked by summed BERT-based relevance to clues.
    """
    clue_texts = {row["number_direction"]: row["clue"] for _, row in cw.clue_df.iterrows()}
    ranked = []

    for solution in solutions:
        total_score = 0
        for var, word in solution.items():
            clue = clue_texts.get(var, "")
            inputs = tokenizer(f"{clue} [SEP] {word}", return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                score = torch.softmax(outputs.logits, dim=1)[0][1].item()  # Relevance score
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
                probs = torch.softmax(outputs.logits, dim=1)[:, 1].tolist()

            scores.extend(probs)

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("⚠️ CUDA OOM. Try reducing batch size.")
                torch.cuda.empty_cache()
                raise e
            else:
                raise e
    return scores


def generate_variables_domains_constraints_from_crossword_ranked(cw, tokenizer, model, top_k=5, device="cuda"):
    from clue_solving.letter_pattern_matching import find_words_by_pattern
    from clue_solving.csp_pattern_matching import get_filled_words
    from clue_solving.BERT_similarity_ranking import rank_candidates_batch

    import traceback
    print("🔁 generate_variables_domains_constraints_from_crossword_ranked() CALLED FROM:")
    traceback.print_stack(limit=5)


    clues = cw.clue_df.to_dict(orient="records")
    grid_map = defaultdict(list)
    variables = []
    raw_domains = {}
    final_domains = {}
    constraints = set()
    grid = cw.grid

    var_to_positions = {}
    var_to_clue = {}
    filled = get_filled_words(cw)

    print("📦 Generating variables and initial domains (constraint-aware)...")

    for clue in tqdm(clues, desc="🧩 Processing clues"):
        var = clue["number_direction"]
        clue_text = clue["clue"]
        start_row, start_col = clue["start_row"], clue["start_col"]
        end_row, end_col = clue["end_row"], clue["end_col"]
        length = clue["length"]

        # Determine clue positions
        if start_row == end_row:
            positions = [(start_row, start_col + i) for i in range(length)]
        elif start_col == end_col:
            positions = [(start_row + i, start_col) for i in range(length)]
        else:
            raise ValueError(f"Invalid clue shape: {var}")

        variables.append(var)
        var_to_positions[var] = positions
        var_to_clue[var] = clue_text

        for i, (r, c) in enumerate(positions):
            grid_map[(r, c)].append((var, i))

        # If clue is filled, domain is only the filled word
        if var in filled:
            final_domains[var] = [filled[var]]
        else:
            # Build pattern
            pattern = ""
            for (r, c) in positions:
                cell = grid[r][c]
                if isinstance(cell, str) and cell.strip() and len(cell.strip()) == 1 and cell.strip().isalpha():
                    pattern += cell.strip().lower()
                else:
                    pattern += "."


            raw_domains[var] = find_words_by_pattern(pattern)
            print(f"🔍 {var} → pattern '{pattern}' → {len(raw_domains[var])} matches")


    print("📐 Building constraints...")
    for square, entries in grid_map.items():
        if len(entries) > 1:
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    var1, idx1 = entries[i]
                    var2, idx2 = entries[j]
                    constraints.add((var1, var2, idx1, idx2))
                    constraints.add((var2, var1, idx2, idx1))

    print("🔍 Filtering domains using constraints...")
    for var in tqdm(variables, desc="⛓️ Filtering with overlaps"):
        if var in filled:
            continue  # Already assigned

        positions = var_to_positions[var]
        filtered = []

        for word in raw_domains.get(var, []):
            valid = True
            for i, (r, c) in enumerate(positions):
                overlaps = grid_map[(r, c)]
                for other_var, other_idx in overlaps:
                    if other_var == var:
                        continue
                    other_cell = grid[r][c]
                    if cell := grid[r][c]:
                        if isinstance(cell, str) and cell.strip() and len(cell.strip()) == 1 and cell.strip().isalpha():
                            if word[i].lower() != cell.strip().lower():
                                valid = False
                                break
                if not valid:
                    break
            if valid:
                filtered.append(word)

        final_domains[var] = filtered

    # Semantic reranking
    print("🤖 BERT ranking within filtered domains...")
    clue_answer_pairs = []
    var_to_candidates = defaultdict(list)

    for var in variables:
        if var in filled:
            continue
        clue = var_to_clue[var]
        for word in final_domains[var]:
            clue_answer_pairs.append((clue, word))
            var_to_candidates[var].append(word)

    scores = rank_candidates_batch(clue_answer_pairs, tokenizer, model, device=device)

    print("🏁 Ranking and selecting top_k...")
    i = 0
    for var in variables:
        if var in filled:
            continue
        scored = [(word, scores[i + j]) for j, word in enumerate(var_to_candidates[var])]
        i += len(var_to_candidates[var])
        final_domains[var] = [word for word, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]]

    print(f"✅ Done! Variables: {len(variables)} | Constraints: {len(constraints)}")
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

    correct = []
    incorrect = []

    for var, predicted in solution.items():
        predicted_clean = predicted.strip().lower()
        true = true_answers.get(var)

        if true is None:
            print(f"⚠️ No true answer found for {var}")
            continue

        if predicted_clean == true.strip().lower():
            correct.append((var, predicted))
        else:
            incorrect.append((var, predicted, true))

    return correct, incorrect

def print_clues_and_answers(cw):
    """
    Prints the clues and their corresponding answers for a crossword.
    """
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

