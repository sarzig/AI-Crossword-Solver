from multiprocessing import Process, Queue
import multiprocessing
import os
import re
import time
import pandas as pd
from tqdm import tqdm
from clue_classification_and_processing.helpers import get_project_root
from clue_solving.letter_pattern_matching import find_words_by_pattern
from puzzle_objects.crossword_and_clue import Crossword

multiprocessing.set_start_method("spawn", force=True)

import nltk
nltk.download('words')

def is_valid(assignment, constraints):
    
    for i1, i2, pos1, pos2 in constraints:
        if i1 in assignment and i2 in assignment:
            w1, w2 = assignment[i1], assignment[i2]
            if len(w1) <= pos1 or len(w2) <= pos2:
                return False

            c1, c2 = w1[pos1], w2[pos2]
            
            if c1.lower() != c2.lower():
                return False

    return True

def compute_letter_frequencies(words):
    letter_counts = Counter()
    total = 0
    for word in words:
        for ch in word.lower():
            if ch.isalpha():
                letter_counts[ch] += 1
                total += 1
    freq = {ch: round((count / total) * 100, 2) for ch, count in letter_counts.items()}
    return freq

def word_score(word, freq_table):
    return sum(freq_table.get(ch.lower(), 0) for ch in word)


# def solve_csp(variables, domains, constraints, assignment={}):
#     if len(assignment) == len(variables):
#         return [assignment.copy()]

#     solutions = []
#     var = [v for v in variables if v not in assignment][0]

#     for value in domains[var]:
#         assignment[var] = value
#         if is_valid(assignment, constraints):
#             solutions.extend(solve_csp(variables, domains, constraints, assignment))
#         del assignment[var]

#     return solutions

### 
# Partial CSP
###


# def solve_all_max_partial_csp(variables, domains, constraints, assignment=None):
#     from collections import defaultdict

#     if assignment is None:
#         assignment = {}

#     var_constraints = defaultdict(list)
#     for v1, v2, i1, i2 in constraints:
#         var_constraints[v1].append((v1, v2, i1, i2))
#         var_constraints[v2].append((v2, v1, i2, i1))

#     sorted_vars = sorted(
#         [v for v in variables if domains[v] and v not in assignment],
#         key=lambda v: -len(var_constraints[v])
#     )

#     all_solutions = []
#     max_len = [len(assignment)]

#     def backtrack(current_assignment, remaining_vars):
#         if not is_valid(current_assignment, constraints):
#             return

#         current_len = len(current_assignment)
#         if current_len > max_len[0]:
#             max_len[0] = current_len
#             all_solutions.clear()
#             all_solutions.append(current_assignment.copy())
#         elif current_len == max_len[0]:
#             all_solutions.append(current_assignment.copy())

#         if not remaining_vars:
#             return

#         next_var = min(remaining_vars, key=lambda v: len(domains[v]))
#         # for value in domains[next_var]:
#         for value in tqdm(domains[next_var], desc=f"Trying {next_var}", leave=False):
#             current_assignment[next_var] = value
#             backtrack(current_assignment, [v for v in remaining_vars if v != next_var])
#             del current_assignment[next_var]

#     # remaining_vars = [v for v in variables if domains[v] and v not in assignment]
#     backtrack(assignment.copy(), sorted_vars)
#     return all_solutions

# With letter frequency
def solve_all_max_partial_csp(variables, domains, constraints, assignment=None, freq_table=None, tqdm_enabled=True):
    if assignment is None:
        assignment = {}

    var_constraints = defaultdict(list)
    for v1, v2, i1, i2 in constraints:
        var_constraints[v1].append((v1, v2, i1, i2))
        var_constraints[v2].append((v2, v1, i2, i1))

    sorted_vars = sorted(
        [v for v in variables if domains[v] and v not in assignment],
        key=lambda v: -len(var_constraints[v])
    )

    all_solutions = []
    max_len = [len(assignment)]

    def backtrack(current_assignment, remaining_vars):
        if not is_valid(current_assignment, constraints):
            return

        current_len = len(current_assignment)
        if current_len > max_len[0]:
            max_len[0] = current_len
            all_solutions.clear()
            all_solutions.append(current_assignment.copy())
        elif current_len == max_len[0]:
            all_solutions.append(current_assignment.copy())

        if not remaining_vars:
            return

        next_var = min(remaining_vars, key=lambda v: len(domains[v]))
        candidates = domains[next_var]
        if freq_table:
            candidates = sorted(candidates, key=lambda w: -word_score(w, freq_table))

        # for value in tqdm(candidates, desc=f"Trying {next_var}", leave=False):
        #     current_assignment[next_var] = value
        #     backtrack(current_assignment, [v for v in remaining_vars if v != next_var])
        #     del current_assignment[next_var]

        iterator = tqdm(candidates, desc=f"Trying {next_var}", leave=False) if tqdm_enabled else candidates

        for value in iterator:
            current_assignment[next_var] = value
            backtrack(current_assignment, [v for v in remaining_vars if v != next_var])
            del current_assignment[next_var]

    backtrack(assignment.copy(), sorted_vars)
    return all_solutions


# def find_all_valid_full_extensions(base_assignment, variables, domains, constraints, verbose=False):
#     from collections import defaultdict

#     results = []

#     # Build constraint lookup
#     constraint_map = defaultdict(list)
#     for v1, v2, i1, i2 in constraints:
#         constraint_map[v1].append((v1, v2, i1, i2))
#         constraint_map[v2].append((v2, v1, i2, i1))

#     def is_partial_valid(assign):
#         for v1 in assign:
#             for (a, b, i, j) in constraint_map[v1]:
#                 if b in assign:
#                     if assign[a][i].lower() != assign[b][j].lower():
#                         if verbose:
#                             print(f"Constraint fail: {a}[{i}]={assign[a][i]} != {b}[{j}]={assign[b][j]}")
#                         return False
#         return True

#     def backtrack(curr_assign):
#         if not is_partial_valid(curr_assign):
#             if verbose:
#                 print(f"Pruned: {curr_assign}")
#             return

#         if len(curr_assign) == len(variables):
#             if verbose:
#                 print(f"✅ Found full valid extension: {curr_assign}")
#             results.append(curr_assign.copy())
#             return

#         unassigned = [v for v in variables if v not in curr_assign and domains[v]]
#         if not unassigned:
#             return

#         next_var = min(unassigned, key=lambda v: len(domains[v]))
#         for value in domains[next_var]:
#             curr_assign[next_var] = value
#             if verbose:
#                 print(f"Trying {next_var} = {value}")
#             backtrack(curr_assign)
#             del curr_assign[next_var]

#     backtrack(base_assignment.copy())
#     return results



def find_all_valid_full_extensions(base_assignment, variables, domains, constraints, verbose=False, freq_table=None):
    from collections import defaultdict
    from tqdm import tqdm

    results = []
    constraint_map = defaultdict(list)
    for v1, v2, i1, i2 in constraints:
        constraint_map[v1].append((v1, v2, i1, i2))
        constraint_map[v2].append((v2, v1, i2, i1))

    def is_partial_valid(assign):
        for v1 in assign:
            for (a, b, i, j) in constraint_map[v1]:
                if b in assign:
                    if assign[a][i].lower() != assign[b][j].lower():
                        if verbose:
                            print(f"Constraint fail: {a}[{i}]={assign[a][i]} != {b}[{j}]={assign[b][j]}")
                        return False
        return True

    def word_score(word, freq_table):
        return sum(freq_table.get(ch.lower(), 0) for ch in word)

    def backtrack(curr_assign):
        if not is_partial_valid(curr_assign):
            if verbose:
                print(f"Pruned: {curr_assign}")
            return

        if len(curr_assign) == len(variables):
            if verbose:
                print(f"✅ Found full valid extension: {curr_assign}")
            results.append(curr_assign.copy())
            return

        unassigned = [v for v in variables if v not in curr_assign and domains[v]]
        if not unassigned:
            return

        next_var = min(unassigned, key=lambda v: len(domains[v]))
        candidates = domains[next_var]
        if freq_table:
            candidates = sorted(candidates, key=lambda w: -word_score(w, freq_table))

        for value in tqdm(candidates, desc=f"Extending {next_var}", leave=False) if verbose else candidates:
            curr_assign[next_var] = value
            backtrack(curr_assign)
            del curr_assign[next_var]

    backtrack(base_assignment.copy())
    return results

# Solve CSP
def solve_in_two_phases(variables, domains, constraints, domain_threshold=100, verbose=True, freq_table=None, assignment=None):
    from tqdm import tqdm

    # 1. Split variables
    small_domain_vars = [v for v in variables if len(domains[v]) <= domain_threshold]
    large_domain_vars = [v for v in variables if len(domains[v]) > domain_threshold]

    if verbose:
        print(f"Running CSP on {len(small_domain_vars)} small-domain variables first...")
        for v in large_domain_vars:
            print(f"⏭️ Deferring {v} (domain size = {len(domains[v])})")

    # Filter constraints that only involve small domain vars
    filtered_constraints = [c for c in constraints if c[0] in small_domain_vars and c[1] in small_domain_vars]

    # 2. Solve CSP on small domain variables
    base_solutions = solve_all_max_partial_csp(small_domain_vars, domains, filtered_constraints, freq_table=freq_table, assignment=assignment, tqdm_enabled=True)

    if not base_solutions:
        print("❌ No base partial solutions found.")
        return []

    all_extensions = []
    for i, base in enumerate(base_solutions):
        if verbose:
            print(f"🧹 Max partial {i+1} from Phase 1:")
            for k in sorted(base):
                print(f"  {k}: {base[k]}")

        if len(base) == len(variables):
            print("✅ Partial is already complete! No need to extend.")
            all_extensions.append(base)
            continue

        print(f"🔁 Extending base partial {i+1}...")
        extensions = find_all_valid_full_extensions(base, variables, domains, constraints, verbose=False, freq_table=freq_table)

        if not extensions:
            print("❌ No valid extensions found from this base.")
        else:
            print(f"✅ Found {len(extensions)} valid full extensions from this base.")
            all_extensions.extend(extensions)

        # Auto-place if only one solution
    if len(all_extensions) == 1:
        print("🔧 Placing words from the unique solution into the crossword...")
        solution = all_extensions[0]
        for var, word in solution.items():
            try:
                cw.place_word(word, var)
                print(f"  ✅ Placed {word} in {var}")
            except Exception as e:
                print(f"  ❌ Could not place {word} in {var}: {e}")

    return all_extensions

#########################################################################################
# Generate variables, domains, constraints 
#########################################################################################

from collections import Counter, defaultdict

def generate_variables_domains_constraints_from_crossword(crossword_object):
    clues = crossword_object.clue_df.to_dict(orient="records")

    from collections import defaultdict
    grid_map = defaultdict(list)  # (row, col) → list of (var_id, letter_index)

    variables = []
    domains = {}
    constraints = set()
    grid = crossword_object.grid

    for clue in clues:
        var_name = clue["number_direction"]
        start_row = clue["start_row"]
        start_col = clue["start_col"]
        end_row = clue["end_row"]
        end_col = clue["end_col"]
        length = clue["length"]

        # Determine positions the clue touches
        positions = []
        if start_row == end_row:  # Across
            positions = [(start_row, start_col + i) for i in range(length)]
        elif start_col == end_col:  # Down
            positions = [(start_row + i, start_col) for i in range(length)]
        else:
            raise ValueError(f"Clue {var_name} is not strictly across or down")

        # Build pattern from grid contents
        pattern = ""
        for (r, c) in positions:
            cell = grid[r][c]
            if cell.startswith("[") and cell.endswith("]") and len(cell) == 3:
                ch = cell[1]
                pattern += ch if ch != " " else "."
            else:
                pattern += "."

        variables.append(var_name)

        # Use exact word if known, otherwise generate candidates
        if "." not in pattern:
            domains[var_name] = [pattern]
        else:
            domains[var_name] = find_words_by_pattern(pattern)

        # Add position mapping for constraint generation
        for i, (r, c) in enumerate(positions):
            grid_map[(r, c)].append((var_name, i))

    # Generate constraints for overlapping positions
    for square, entries in grid_map.items():
        if len(entries) > 1:
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    var1, idx1 = entries[i]
                    var2, idx2 = entries[j]
                    constraints.add((var1, var2, idx1, idx2))
                    constraints.add((var2, var1, idx2, idx1))

    return variables, domains, list(constraints)

def generate_variables_domains_constraints_from_file(crossword_file):
    if crossword_file.endswith(".csv"):
        df = pd.read_csv(crossword_file)
    elif crossword_file.endswith(".xlsx"):
        df = pd.read_excel(crossword_file)
    else:
        raise ValueError("File must be .csv or .xlsx")

    cw = Crossword(clue_df=df)
    return generate_variables_domains_constraints_from_crossword(cw)

# def get_filled_words(cw):
#     filled = {}
#     for _, row in cw.clue_df.iterrows():
#         var = row["number_direction"]
#         word = []

#         start_row, start_col = row["start_row"], row["start_col"]
#         end_row, end_col = row["end_row"], row["end_col"]

#         if start_row == end_row:  # Across
#             for c in range(start_col, end_col + 1):
#                 cell = cw.grid[start_row][c]
#                 word.append(cell[1] if cell != "[ ]" else ".")
#         elif start_col == end_col:  # Down
#             for r in range(start_row, end_row + 1):
#                 cell = cw.grid[r][start_col]
#                 word.append(cell[1] if cell != "[ ]" else ".")
#         else:
#             continue

#         filled_word = "".join(word)
#         if "." not in filled_word:
#             filled[var] = filled_word

#     return filled

def get_filled_words(cw):
    filled = {}
    for _, row in cw.clue_df.iterrows():
        var = row["number_direction"]
        word = []

        start_row, start_col = row["start_row"], row["start_col"]
        end_row, end_col = row["end_row"], row["end_col"]

        if start_row == end_row:  # Across
            for c in range(start_col, end_col + 1):
                cell = cw.grid[start_row][c]
                if isinstance(cell, str) and cell.strip() and len(cell.strip()) == 1 and cell.strip().isalpha():
                    word.append(cell.strip())
                else:
                    word.append(".")
        elif start_col == end_col:  # Down
            for r in range(start_row, end_row + 1):
                cell = cw.grid[r][start_col]
                if isinstance(cell, str) and cell.strip() and len(cell.strip()) == 1 and cell.strip().isalpha():
                    word.append(cell.strip())
                else:
                    word.append(".")

        else:
            continue

        filled_word = "".join(word)
        if "." not in filled_word:
            filled[var] = filled_word

    return filled


def load_crossword_from_file(file_name):

    if file_name.endswith(".csv"):
        df = pd.read_csv(file_name)
    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(file_name)
    else:
        raise ValueError("File must be .csv or .xlsx")

    cw = Crossword(clue_df=df)

    enriched_datafram = cw.clue_df
    print(enriched_datafram)

    return cw

def load_crossword_from_file_with_answers(file_name):
    if file_name.endswith(".csv"):
        df = pd.read_csv(file_name)
    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(file_name)
    else:
        raise ValueError("File must be .csv or .xlsx")

    # Normalize column names
    df.columns = [col.strip().lower() for col in df.columns]

    # Rename known optional columns if they exist
    if "answer (optional column, for checking only)" in df.columns:
        df.rename(columns={"answer (optional column, for checking only)": "answer"}, inplace=True)

    if "length (optional column, for checking only)" in df.columns:
        df.rename(columns={"length (optional column, for checking only)": "length"}, inplace=True)

    # Pass cleaned DataFrame into Crossword
    cw = Crossword(clue_df=df)

    print(cw.clue_df.head())  # Confirm column structure and content
    return cw

#############################################################
# Load cross word file
#############################################################
# mini_loc = f"{get_project_root()}/data/puzzle_samples/mini_03262025.xlsx"

# cw = load_crossword_from_file(mini_loc)

# # Non existing words
# cw.place_word("irl", "4-Across")
# cw.place_word("paris", "5-Across")
# cw.place_word("arroz", "2-Down")
# cw.place_word("pbj", "5-Down")

# # # Regular words
# # cw.place_word("tab", "1-Across")
# # cw.place_word("broth", "7-Across")
# # cw.place_word("jazzy", "8-Across")
# # cw.place_word("blitz", "3-Down")
# # cw.place_word("shy", "6-Down")

# cw.detailed_print()


# variables, domains, constraints = generate_variables_domains_constraints_from_crossword(cw)

# print(f"variables: {variables}")
# print(f"domains: {domains}")
# print(f"constraints: {constraints}")

# filled = get_filled_words(cw)

# # from nltk.corpus import words  # assuming you've already downloaded `words`
# # word_list = set(word.lower() for word in words.words())
# # # Compute letter frequencies from the domain words only
# # all_domain_words = [word for domain in domains.values() for word in domain]
# # freq_table = compute_letter_frequencies(all_domain_words)

# # # Proceed to solve CSP
# # # partial_solutions = solve_all_max_partial_csp(variables, domains, constraints, assignment=filled, freq_table=freq_table)

# partial_solutions = solve_all_max_partial_csp(variables, domains, constraints, assignment=filled)

# print(f"partial solutions: {partial_solutions}")


# for var, domain in domains.items():
#     if var not in filled:
#         print(f"{var} has {len(domain)} options → {domain}...")

# # Get filled entries
# filled = get_filled_words(cw)

# base_solutions = solve_all_max_partial_csp(variables, domains, constraints)

# if not base_solutions:
#     print("No base partials found.")
# else:
#     base = base_solutions[0]  # choose first max-length base
#     print("🧩 Base max-length partial 1:")
#     for k in sorted(base):
#         print(f"  {k}: {base[k]}")

#     if len(base) < len(variables):
#         extensions = find_all_valid_full_extensions(base, variables, domains, constraints, verbose=True)
#         print(f"\n🔍 Valid extensions from this base (found {len(extensions)}):")
#         for i, ext in enumerate(extensions, 1):
#             print(f"  ➕ Extension {i}:")
#             for k in sorted(ext):
#                 print(f"    {k}: {ext[k]}")
#     else:
#         print("✅ Base is already a full valid solution — no need to extend.")

