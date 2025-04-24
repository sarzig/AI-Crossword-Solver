from tqdm import tqdm
from clue_classification_and_processing.helpers import get_project_root
from clue_solving.csp_pattern_matching import load_crossword_from_file_with_answers
 
def lowercase_solutions(solutions):
    return [{k: v.lower() if isinstance(v, str) else v for k, v in sol.items()} for sol in solutions]

def get_overlapping_clues(subset1_solutions, subset2_solutions):

    if not subset1_solutions or not subset2_solutions:
        return []
    

    vars1 = set(subset1_solutions[0].keys())
    vars2 = set(subset2_solutions[0].keys())

    return sorted(vars1 & vars2)

 
def get_common_values(solutions1, solutions2, clue_key):
    values1 = {sol_dict.get(clue_key) for sol_dict in solutions1}
    values2 = {sol_dict.get(clue_key) for sol_dict in solutions2}
    return values1 & values2

 
def filter_solutions_by_value(solutions, clue_key, allowed_values):
    return [sol_dict for sol_dict in solutions if sol_dict.get(clue_key) in allowed_values]

 
def prune_on_multiple_overlaps(subset1, subset2, overlapping_clues):
    for clue_key in overlapping_clues:
        common_values = get_common_values(subset1, subset2, clue_key)
        if not common_values:
            return [], []
        subset1 = filter_solutions_by_value(subset1, clue_key, common_values)
        subset2 = filter_solutions_by_value(subset2, clue_key, common_values)
        if not subset1 or not subset2:
            return [], []
    return subset1, subset2

 
def merge_matching_solutions(subset1, subset2):

    if subset1 is None or subset2 is None:
        return []

    # vars1 = set(subset1.clue_df["number_direction"])
    # vars2 = set(subset2.clue_df["number_direction"])

    overlapping_clues = list(set(subset1[0].keys()) & set(subset2[0].keys()))

    merged = []
    for sol1 in subset1:
        for sol2 in subset2:
            if all(sol1.get(clue) == sol2.get(clue) for clue in overlapping_clues):
                merged.append({**sol1, **sol2})
    unique_merged = [dict(t) for t in {tuple(sorted(d.items())) for d in merged}]
    return unique_merged


def deduplicate_solutions(solutions):
    seen = set()
    unique = []
    for sol in solutions:
        key = tuple(sorted(sol.items()))
        if key not in seen:
            seen.add(key)
            unique.append(sol)
        if len(unique) >= MAX_MERGED_SOLUTIONS:
            print(f" Deduplication cap of {MAX_MERGED_SOLUTIONS} reached, truncating results.")
            break
    return unique


MAX_MERGED_SOLUTIONS = 10000  # adjust as needed

def merge_matching_solution_sets(set1, set2, overlap_threshold=1):
    merged = []
    for s1 in set1:
        for s2 in set2:
            # Skip if not enough overlap to matter
            overlap = set(s1.keys()) & set(s2.keys())
            if len(overlap) < overlap_threshold:
                continue

            # Only merge if no conflicts
            if all(s1.get(k) == s2.get(k) for k in overlap):
                combined = {**s1, **s2}
                merged.append(combined)
                if len(merged) >= MAX_MERGED_SOLUTIONS:
                    print(f"Merge cap of {MAX_MERGED_SOLUTIONS} reached, stopping early.")
                    return deduplicate_solutions(merged)

    return deduplicate_solutions(merged)
########################################################################################
# Example Usage
########################################################################################
# subset1_pruned, subset2_pruned = prune_on_multiple_overlaps(
#     subset1_solutions,
#     subset2_solutions,
#     overlapping_clues
# )
 
# print(f"subset1_pruned: {subset1_pruned}")
# print(f"subset2_pruned: {subset2_pruned}")
 
# [({'1-Across': 'MATH', '3-Across': 'CAT', '1-Down': 'COW'}, 9.2),
#  ({'1-Across': 'MATH', '3-Across': 'DOG', '1-Down': 'DAY'}, 8.8),
#  ({'1-Across': 'MATH', '3-Across': 'CAR', '1-Down': 'CAP'}, 7.9)]
 
# [({'3-Across': 'CAT', '4-Across': 'RIDE', '1-Down': 'COW'}, 9.1),
#  ({'3-Across': 'DOG', '4-Across': 'WALK', '1-Down': 'DAY'}, 8.7),
#  ({'3-Across': 'CAR', '4-Across': 'ZOOM', '1-Down': 'CAP'}, 7.8)]
 
 



 