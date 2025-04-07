#########################################################################################
# Placing answers that are not in the word list (for testing)
# #########################################################################################

from multiprocessing import Process, Queue
from nltk.corpus import words
import os
import re

from clue_classification_and_processing.helpers import get_project_root
from clue_solving.csp_pattern_matching import compute_letter_frequencies, generate_variables_domains_constraints_from_crossword, get_filled_words, load_crossword_from_file_with_answers, solve_in_two_phases

def auto_place_non_dict_words(cw, vocab_path="combined_vocab.txt", verbose=True):
    # Load the combined vocab from file
    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            word_list = set(line.strip().lower() for line in f if line.strip())
    except FileNotFoundError:
        print(f"Could not find vocab file: {vocab_path}")
        return

    # Identify the answer column
    answer_col_name = None
    for col in cw.clue_df.columns:
        if "answer" in col.lower():
            answer_col_name = col
            break

    if not answer_col_name:
        print("No answer column found in clue_df.")
        return

    non_dict_words = []

    for _, row in cw.clue_df.iterrows():
        var = row["number_direction"]
        answer = str(row.get(answer_col_name, "")).strip().lower()

        if not answer:
            continue

        if answer not in word_list:
            try:
                cw.place_word(answer, var)
                non_dict_words.append((var, answer))
                if verbose:
                    print(f"✅ Placed non-dictionary word: cw.place_word('{answer}', '{var}')")
            except Exception as e:
                print(f"❌ Could not place '{answer}' at '{var}': {e}")

    if not non_dict_words:
        print("No non-dictionary answers found.")
    else:
        print(f"\n Placed {len(non_dict_words)} non-dictionary words:")
        for var, word in non_dict_words:
            print(f'cw.place_word("{word}", "{var}")')


#########################################################################################
# Auto Testing -  With words that don't exist in the dictionary already placed
#########################################################################################


# mini_loc = f"{get_project_root()}/data/puzzle_samples/mini_03262025.xlsx"
# cw = load_crossword_from_file_with_answers(mini_loc)

# # Automatically detect and place non-dictionary words
# auto_place_non_dict_words(cw)

# # Optional: visually verify
# cw.detailed_print()

# variables, domains, constraints = generate_variables_domains_constraints_from_crossword(cw)


# filled = get_filled_words(cw)

# partial_solutions = solve_all_max_partial_csp(variables, domains, constraints, assignment=filled)

# print(f"partial solutions: {partial_solutions}")
    
#########################################################################################
# Testing - clueset 2
#########################################################################################


# mini_loc = f"{get_project_root()}/data/puzzle_samples/processed_puzzle_samples/mini_2024_03_17.csv"

# cw = load_crossword_from_file_with_answers(mini_loc)

# # Automatically detect and place non-dictionary words
# auto_place_non_dict_words(cw)

# # Optional: visually verify
# cw.detailed_print()

# variables, domains, constraints = generate_variables_domains_constraints_from_crossword(cw)


# filled = get_filled_words(cw)

# partial_solutions = solve_all_max_partial_csp(variables, domains, constraints, assignment=filled)

# print(f"partial solutions: {partial_solutions}")

#########################################################################################
# Testing - clueset 2 - this one takes a really really long time and literally returns 200,000 different options
#########################################################################################

# mini_loc = f"{get_project_root()}/data/puzzle_samples/processed_puzzle_samples/mini_2024_03_02.csv"

# cw = load_crossword_from_file_with_answers(mini_loc)

# # Automatically detect and place non-dictionary words
# auto_place_non_dict_words(cw)

# # Optional: visually verify
# cw.detailed_print()

# variables, domains, constraints = generate_variables_domains_constraints_from_crossword(cw)

# # for var, domain in domains.items():
# #     if var not in filled:
# #         print(f"\n{var} has {len(domain)} options → {domain}...\n")

# for var, domain in domains.items():
#     if var not in filled:
#         print(f"{var} has {len(domain)}")

# filled = get_filled_words(cw)

# # # partial_solutions = solve_all_max_partial_csp(variables, domains, constraints, assignment=filled)
# # partial_solutions = solve_all_max_partial_csp(
# #     variables, domains, constraints, assignment=filled
# # )

# from nltk.corpus import words  # assuming you've already downloaded `words`
# word_list = set(word.lower() for word in words.words())
# # Compute letter frequencies from the domain words only
# all_domain_words = [word for domain in domains.values() for word in domain]
# freq_table = compute_letter_frequencies(all_domain_words)

# # # Proceed to solve CSP
# # partial_solutions = solve_all_max_partial_csp(variables, domains, constraints, assignment=filled, freq_table=freq_table)



# # print(f"partial solutions: {partial_solutions}")


# solutions = solve_in_two_phases(
#     variables,
#     domains,
#     constraints,
#     domain_threshold=100,
#     verbose=True,
#     freq_table=freq_table,
#     assignment=filled  # ✅ Add this!
# )

# if solutions:
#     print("🎉 Final solution:")
#     for k in sorted(solutions[0]):
#         print(f"  {k}: {solutions[0][k]}")
# else:
#     print("😞 No solution found.")


#########################################################################################
# Testing - clueset 4
#########################################################################################


# mini_loc = f"{get_project_root()}/data/puzzle_samples/processed_puzzle_samples/mini_2024_03_18.csv"

# cw = load_crossword_from_file_with_answers(mini_loc)

# # Automatically detect and place non-dictionary words
# auto_place_non_dict_words(cw)

# # Optional: visually verify
# cw.detailed_print()

# variables, domains, constraints = generate_variables_domains_constraints_from_crossword(cw)


# filled = get_filled_words(cw)

# # partial_solutions = solve_all_max_partial_csp(variables, domains, constraints, assignment=filled)
# partial_solutions = solve_in_two_phases(variables, domains, constraints, domain_threshold=100, verbose=True, freq_table=None, assignment=filled)

# cw.detailed_print()

# print(f"partial solutions: {partial_solutions}")


#########################################################################################
# Testing - loop to test all minis
#########################################################################################
# safe solver
def safe_solve(variables, domains, constraints, filled, freq_table, queue):
    import time
    try:
        start = time.time()
        solutions = solve_in_two_phases(
            variables, domains, constraints,
            domain_threshold=100,
            verbose=False,
            freq_table=freq_table,
            assignment=filled
        )
        duration = time.time() - start

        queue.put(("ok", len(solutions), duration, solutions[0] if solutions else None))

    except Exception as e:
        queue.put(("error", str(e)))


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

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


    # each file in a subprocess with a 5-minute timeout
    TIMEOUT = 180  # seconds

    from pathlib import Path
    import json
    from datetime import datetime

    FAILED_LOG_PATH = "failed_csp_logs_solutions_less_than_1000_3_minutes.json"
    SUCCESS_LOG_PATH = "successful_csp_logs_solutions_less_than_1000_3_minutes.json"

    # Load or initialize logs
    def load_log(file_path):
        if Path(file_path).exists():
            return json.loads(Path(file_path).read_text())
        return {}

    failed_logs = load_log(FAILED_LOG_PATH)
    success_logs = load_log(SUCCESS_LOG_PATH)

    puzzle_dir = f"{get_project_root()}/data/puzzle_samples/processed_puzzle_samples"


    for name in mini_files_to_run:
        for ext in [".csv", ".xlsx"]:
            file_path = os.path.join(puzzle_dir, name + ext)
            if not os.path.exists(file_path):
                continue

            print(f"\n🚀 Solving {name}{ext}...")

            try:
                # 💡 Do prep outside the subprocess so we can retain domain_lengths
                cw = load_crossword_from_file_with_answers(file_path)
                auto_place_non_dict_words(cw)
                variables, domains, constraints = generate_variables_domains_constraints_from_crossword(cw)
                filled = get_filled_words(cw)
                all_domain_words = [word for domain in domains.values() for word in domain]
                freq_table = compute_letter_frequencies(all_domain_words)
                domain_lengths = {var: len(domain) for var, domain in domains.items()}
            except Exception as e:
                print(f"❌ Preprocessing error: {e}")
                failed_logs[file_path] = {
                    "status": "prep_error",
                    "error_message": str(e),
                    "domain_lengths": {},
                    "timestamp": datetime.now().isoformat()
                }
                continue

            queue = Queue()
            p = Process(target=safe_solve, args=(variables, domains, constraints, filled, freq_table, queue))
            p.start()
            p.join(timeout=TIMEOUT)

            if p.is_alive():
                p.terminate()
                p.join()
                print(f"⏱️ Timeout! Took more than {TIMEOUT / 60:.1f} minutes.")

                failed_logs[file_path] = {
                    "status": "timeout",
                    "domain_lengths": domain_lengths,
                    "timestamp": datetime.now().isoformat()
                }

            elif not queue.empty():
                result = queue.get()
                if result[0] == "error":
                    print(f"❌ Solver error: {result[1]}")
                    failed_logs[file_path] = {
                        "status": "error",
                        "error_message": result[1],
                        "domain_lengths": domain_lengths,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    _, num_solutions, duration, solution_data = result
                    print(f"✅ Solutions: {num_solutions} | Time: {duration:.2f}s")

                    if num_solutions <= 1000:
                        print(f"🧩 Logging {num_solutions} solution(s) into success log.")
                        all_solutions = result[3] if isinstance(result[3], list) else [result[3]]

                        success_logs[file_path] = {
                            "status": "success",
                            "num_solutions": num_solutions,
                            "duration_sec": duration,
                            "domain_lengths": domain_lengths,
                            "timestamp": datetime.now().isoformat(),
                            "solutions": all_solutions,
                        }

                        # Optional: place the first solution into the crossword for display
                        first_solution = all_solutions[0]
                        for k, word in first_solution.items():
                            try:
                                cw.place_word(word, k)
                            except Exception as e:
                                print(f"Failed to place {word} for {k}: {e}")
                        cw.detailed_print()
                    else:
                        print(f"ℹ️ Skipping log — {num_solutions} solutions is more than 10.")
            else:
                print("❌ No result returned by solver.")
                failed_logs[file_path] = {
                    "status": "no_result",
                    "domain_lengths": domain_lengths,
                    "timestamp": datetime.now().isoformat()
                }

    with open(FAILED_LOG_PATH, "w") as f:
        json.dump(failed_logs, f, indent=2)

    with open(SUCCESS_LOG_PATH, "w") as f:
        json.dump(success_logs, f, indent=2)

    print(f"\n📄 Logged {len(success_logs)} successes to {SUCCESS_LOG_PATH}")
    print(f"📄 Logged {len(failed_logs)} failures to {FAILED_LOG_PATH}")
