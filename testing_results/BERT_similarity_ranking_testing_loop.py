from multiprocessing import Process, Queue
import os
from clue_classification_and_processing.helpers import get_project_root
from clue_solving.BERT_similarity_ranking import generate_variables_domains_constraints_from_crossword_ranked, get_true_answers, rank_csp_solutions
from clue_solving.csp_pattern_matching import load_crossword_from_file_with_answers, get_filled_words, solve_in_two_phases
import torch
import json
from datetime import datetime

# Load BERT reranker
from transformers import AutoTokenizer, AutoModelForSequenceClassification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-small").to(device)
model.eval()

PUZZLE_DIR = f"{get_project_root()}/data/puzzle_samples/processed_puzzle_samples"
LOG_FILE = "bert_rerank_eval_log.json"

def safe_run(file_path, queue):
    try:
        cw = load_crossword_from_file_with_answers(file_path)
        cw.place_helper_answers(fill_percent=60)
        filled = get_filled_words(cw)
        variables, domains, constraints = generate_variables_domains_constraints_from_crossword_ranked(
            cw, tokenizer, model, top_k=2000, device=device
        )
        solutions = solve_in_two_phases(
            variables, domains, constraints,
            domain_threshold=100,
            assignment=filled,
            verbose=False
        )
        true_cleaned = {k: v.strip().lower() for k, v in get_true_answers(cw).items()}

        match_found = False
        for sol in solutions:
            pred = {k: v.strip().lower() for k, v in sol.items()}
            if pred == true_cleaned:
                match_found = True
                break
        
        top_ranked_solution = {}
        top_ranked_match = False
        num_correct = 0

        if solutions:
            ranked = rank_csp_solutions(cw, solutions, tokenizer, model, device)
            top_ranked_solution = {k: v.strip().lower() for k, v in ranked[0][0].items()}
            top_ranked_match = top_ranked_solution == true_cleaned
            num_correct = sum(1 for k, v in top_ranked_solution.items() if true_cleaned.get(k) == v)


        queue.put({
            "status": "success",
            "match_found": match_found,
            "num_solutions": len(solutions),
            "top_ranked_match": top_ranked_match,
            "num_correct_in_top_ranked": num_correct,
            "top_ranked_solution": top_ranked_solution
        })

    except Exception as e:
        queue.put({
            "status": "error",
            "error": str(e)
        })

def run_batch(mini_filenames):
    logs = {}
    for name in mini_filenames:
        for ext in [".csv", ".xlsx"]:
            file_path = os.path.join(PUZZLE_DIR, name + ext)
            if not os.path.exists(file_path):
                continue

            print(f"\n🧠 Evaluating {name}{ext}...")
            queue = Queue()
            p = Process(target=safe_run, args=(file_path, queue))
            p.start()
            p.join(timeout=600)

            if p.is_alive():
                p.terminate()
                p.join()
                print("⏱️ Timeout!")
                logs[file_path] = {
                    "status": "timeout",
                    "timestamp": datetime.now().isoformat()
                }
                continue

            result = queue.get()
            logs[file_path] = {
                **result,
                "timestamp": datetime.now().isoformat()
            }

            if result["status"] == "success":
                print(f"✅ Solutions: {result['num_solutions']} | Match Found: {result['match_found']}")
            else:
                print(f"❌ Error: {result['error']}")

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

    print(f"\n📄 Logs saved to {LOG_FILE}")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    # Subset or full list
    mini_files_to_run = [
        "mini_2024_03_02",
        "mini_2024_03_04",
        "mini_2024_03_05",
        "mini_2024_03_06",
        "mini_2024_03_07",
        "mini_2024_03_08",
        "mini_2024_03_09",
        "mini_2024_03_10",
        "mini_2024_03_12",
        "mini_2024_03_13",
        "mini_2024_03_14",
        "mini_2024_03_15",
        "mini_2024_03_17",
        "mini_2024_03_18",
        "mini_2024_03_19",
        "mini_2024_03_20",
        "mini_2024_03_23",
        "mini_2024_03_24",
        "mini_2024_03_25",
        "mini_2024_03_26",
        "mini_2024_03_28",
        "mini_2024_03_29",
        "mini_2024_03_30",
        "mini_2025_01_01",
        "mini_2025_01_03",
        "mini_2025_01_04",
        "mini_2025_01_05",
        "mini_2025_01_06",
        "mini_2025_01_07",
        "mini_2025_01_08",
        "mini_2025_01_09",
        "mini_2025_01_10",
        "mini_2025_01_11",
        "mini_2025_01_12",
        "mini_2025_01_13",
        "mini_2025_01_14",
        "mini_2025_01_15",
        "mini_2025_01_16",
        "mini_2025_01_17",
        "mini_2025_01_18",
        "mini_2025_01_19",
        "mini_2025_01_20",
        "mini_2025_01_21",
        "mini_2025_01_22",
        "mini_2025_01_23",
        "mini_2025_01_24",
        "mini_2025_01_25",
        "mini_2025_01_26",
        "mini_2025_01_27",
        "mini_2025_01_28",
        "mini_2025_01_29",
        "mini_2025_01_30",
        "mini_2025_01_31",
        "mini_2025_02_01",
        "mini_2025_02_02",
        "mini_2025_02_03",
        "mini_2025_02_04",
        "mini_2025_02_05",
        "mini_2025_02_06",
        "mini_2025_02_07",
        "mini_2025_02_08",
        "mini_2025_02_09",
        "mini_2025_02_10",
        "mini_2025_02_11",
        "mini_2025_02_12",
        "mini_2025_02_13",
        "mini_2025_02_14",
        "mini_2025_02_15",
        "mini_2025_02_16",
        "mini_2025_02_17",
        "mini_2025_02_18",
        "mini_2025_02_19",
        "mini_2025_02_20",
        "mini_2025_02_21",
        "mini_2025_02_23",
        "mini_2025_02_24",
        "mini_2025_02_25",
        "mini_2025_02_26",
        "mini_2025_02_27",
        "mini_2025_02_28",
        "mini_2025_03_01",
        "mini_2025_03_03",
        "mini_2025_03_04",
        "mini_2025_03_05",
        "mini_2025_03_06",
        "mini_2025_03_07",
        "mini_2025_03_08",
        "mini_2025_03_09",
        "mini_2025_03_10",
        "mini_2025_03_12",
        "mini_2025_03_13",
        "mini_2025_03_14",
        "mini_2025_03_15",
        "mini_2025_03_18",
    ]
    run_batch(mini_files_to_run)
