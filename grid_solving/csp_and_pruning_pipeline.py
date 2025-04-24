from collections import Counter
import torch
from tqdm import tqdm
from BERT_models.cluebert_model import ClueBertRanker
from clue_classification_and_processing.helpers import get_project_root
from clue_solving.custom_BERT_ranking import generate_variables_domains_constraints_from_crossword_ranked, get_true_answers, rank_csp_solutions
from clue_solving.csp_pattern_matching import get_filled_words, is_valid, load_crossword_from_file_with_answers, solve_in_two_phases
from grid_solving.pruning_on_overlaps import get_overlapping_clues, lowercase_solutions, merge_matching_solution_sets, merge_matching_solutions, prune_on_multiple_overlaps
from grid_visualization.crossword_visualizer import CrosswordVisualizer
from testing_results.auto_place_clues import auto_place_clues
from transformers import BertTokenizer, BertForSequenceClassification


def solve_crossword_subset(
    cw,
    seed_clue,
    clue_answer_map={},
    tokenizer=None,
    model=None,
    top_k=200,
    verbose=False,
    force_domain_answer=False,
    visualizer=None
):
    subset = cw.subset_crossword(seed_clue, branching_factor=1, overlap_threshold=1, return_type="crossword")

    auto_place_clues(subset, clue_answer_map)
    filled = get_filled_words(subset)

    variables, domains, constraints = generate_variables_domains_constraints_from_crossword_ranked(
        subset, tokenizer=tokenizer, model=model, top_k=top_k, device=next(model.parameters()).device, force_answer=force_domain_answer
    )

    # Check if true answers are present in domains before solving
    true_answers = get_true_answers(subset)
    true_cleaned = {k: v.strip().lower() for k, v in true_answers.items()}
    missing = []
    for var, true_ans in true_cleaned.items():
        domain = domains.get(var, [])
        # if true_ans not in [d.strip().lower() for d in domain]:
        #     missing.append((var, true_ans))

        if true_ans not in [d.strip().lower() for d in domains[var]]:
            print(f"Reinserting true answer '{true_ans}' into domain for {var}")
            domains[var].append(true_ans)

    subset_solutions = solve_in_two_phases(
        cw=cw, variables=variables, domains=domains, constraints=constraints, domain_threshold=100, verbose=False, assignment=filled
    )

    if visualizer:
        for sol in subset_solutions:
            visualizer.place_solution_on_grid(sol)
            time.sleep(1)  # slow down to visually see the step

    return subset, subset_solutions

def merge_and_prune_subsets(subset1, subset2):

    if subset1 is None or subset2 is None:
        return set()
    
    overlap = get_overlapping_clues(subset1, subset2)
    
    subset1 = lowercase_solutions(subset1)
    subset2 = lowercase_solutions(subset2)

    pruned_subset1, pruned_subset2 = prune_on_multiple_overlaps(subset1, subset2, overlap)
    return merge_matching_solutions(pruned_subset1, pruned_subset2)

def check_true_solution_exists(merged_solutions, *subsets):
    print(f"🔎 Checking for ground truth in {len(merged_solutions)} solutions...")
    combined_true = {}
    for s in subsets:
        true = {k: v.lower() for k, v in get_true_answers(s).items()}
        combined_true.update(true)

    for i, sol in enumerate(merged_solutions):
        if all(sol.get(k) == v for k, v in combined_true.items()):
            print(f"Match found in merged_solutions[{i}]: {sol}")
            return True

    print("Correct solution not found in merged_solutions.")
    print(f"Expected: {combined_true}")
    return False

def solve_crossword_worker(queue, cw, root, clue_answer_map, tokenizer, model, top_k, force_domain_answer=False, visualizer=None):
    print(f"🧪 Solving subset rooted at {root} in worker...")
    try:
        subset, solutions = solve_crossword_subset(
            cw, root, clue_answer_map, tokenizer, model, top_k=top_k, verbose=False, force_domain_answer=force_domain_answer, visualizer=visualizer
        )
        queue.put((subset, solutions))
    except Exception:
        queue.put((None, []))


import multiprocessing

def solve_subset_with_timeout(cw, root, clue_answer_map, tokenizer, model, top_k, timeout=30, force_domain_answer=False):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=solve_crossword_worker,
        args=(queue, cw, root, clue_answer_map, tokenizer, model, top_k, force_domain_answer)
    )
    process.start()
    process.join(timeout)

    if process.is_alive():
        process.terminate()
        process.join()
        print(f"⏱️ Timeout: Skipping {root}")
        return None, []

    return queue.get()


def solve_and_merge_subsets(
    cw,
    root_clues: list,
    clue_answer_map: dict,
    tokenizer,
    model,
    top_k,
    verbose=False,
    force_domain_answer=False,
    visualizer=None
):
    all_subsets = []
    all_solutions = []

    for i, root in enumerate(tqdm(root_clues, desc="📦 Solving and merging subsets")):
        print(f"\nSolving subset {i+1}/{len(root_clues)} from root clue: {root}")
        subset, solutions = solve_crossword_subset(
            cw,
            root,
            clue_answer_map=clue_answer_map,
            tokenizer=tokenizer,
            model=model,
            top_k=top_k,
            verbose=verbose,
            force_domain_answer=force_domain_answer,
            visualizer=visualizer
        )
        all_subsets.append(subset)

        if i == 0:
            merged_solutions = lowercase_solutions(solutions)
        else:
            solutions = lowercase_solutions(solutions)
            new_merged = merge_and_prune_subsets(merged_solutions, solutions)
            
            if not new_merged:
                print("Merge would result in 0 solutions. Aborting and returning current progress.")
                return merged_solutions, all_subsets, clue_answer_map # Don't update anything

            merged_solutions = new_merged

        print(f"🔁 Merged solutions so far: {len(merged_solutions)}")

        if len(merged_solutions) == 1:
            single_solution = merged_solutions[0]
            auto_place_clues(cw, single_solution)

            # Update the *original* clue_answer_map by mutating it
            clue_answer_map.update(single_solution)

        all_solutions.append(solutions)

    return merged_solutions, all_subsets, clue_answer_map

from multiprocessing import Process, Queue
import multiprocessing
import time

def solve_crossword_subset_with_queue(cw, root, clue_answer_map, tokenizer, model, top_k, verbose, queue, force_domain_answer=False):
    try:
        subset, solutions = solve_crossword_subset(
            cw,
            root,
            clue_answer_map=clue_answer_map,
            tokenizer=tokenizer,
            model=model,
            top_k=top_k,
            verbose=verbose,
            force_domain_answer=force_domain_answer
        )
        queue.put((subset, solutions))
    except Exception as e:
        queue.put((None, None))

def solve_and_merge_subsets_with_timeout(
    cw,
    root_clues: list,
    clue_answer_map: dict,
    tokenizer,
    model,
    top_k,
    timeout=30,
    verbose=False,
    force_domain_answer=False,
    visualizer=None
):
    all_subsets = []
    all_solutions = []

    for i, root in enumerate(tqdm(root_clues, desc="📦 Solving and merging subsets")):
        print(f"\nSolving subset {i+1}/{len(root_clues)} from root clue: {root}")
        queue = Queue()
        process = Process(
            target=solve_crossword_subset_with_queue,
            args=(cw, root, clue_answer_map, tokenizer, model, top_k, verbose, queue, force_domain_answer)
        )
        process.start()
        process.join(timeout)

        if process.is_alive():
            print(f"Timeout exceeded for root {root}. Terminating process.")
            process.terminate()
            process.join()
            continue

        if not queue.empty():
            subset, solutions = queue.get()
            if visualizer and solutions:
                for sol in solutions:
                    visualizer.place_solution_on_grid(sol)
                    pygame.time.wait(150)

        else:
            print(f"⚠️ No result returned for root {root}. Skipping.")
            continue

        if not subset or not solutions:
            print(f"⚠️ Failed to solve subset for root {root}. Skipping.")
            continue

        all_subsets.append(subset)
        solutions = lowercase_solutions(solutions)

        if i == 0:
            merged_solutions = solutions
        else:
            new_merged = merge_and_prune_subsets(merged_solutions, solutions)
            if not new_merged:
                print("❌ Merge would result in 0 solutions. Aborting and returning current progress.")
                return merged_solutions, all_subsets
            merged_solutions = new_merged

        print(f"Merged solutions so far: {len(merged_solutions)}")
        all_solutions.append(solutions)

    return merged_solutions if 'merged_solutions' in locals() else [], all_subsets, clue_answer_map

def get_remaining_root_clues(cw, used_roots):
    all_vars = set(cw.clue_df["number_direction"])

    # Gather all variables already included in used subsets
    used_vars = set()
    used_clue_sets = []

    for root in used_roots:
        try:
            subset = cw.subset_crossword(root, branching_factor=1, overlap_threshold=1, return_type="crossword")
            clue_set = set(subset.clue_df["number_direction"])
            used_clue_sets.append(clue_set)
            used_vars.update(clue_set)
        except:
            continue  # Skip any failed subset generation

    # Find candidate clues not already used in any subset
    candidates = set()
    for var in all_vars - used_vars:
        try:
            temp_subset = cw.subset_crossword(var, branching_factor=1, overlap_threshold=1, return_type="crossword")
            clue_set = set(temp_subset.clue_df["number_direction"])

            # Only add if it's a novel clue set
            if all(clue_set != used for used in used_clue_sets):
                candidates.add(var)
        except:
            continue

    return sorted(candidates)


def get_low_overlap_candidates(cw, used_roots, overlap_count=(3, 4, 5, 6, 7, 8), current_region_vars=None):
    """
    Find root clues whose generated subsets overlap with the current region via exactly 1 or 2 clues.
    """
    candidates = []
    all_vars = set(cw.clue_df["number_direction"])

    if current_region_vars is None:
        # Compute clue variables already covered
        current_region_vars = set()
        for root in used_roots:
            try:
                subset = cw.subset_crossword(root, branching_factor=1, overlap_threshold=1, return_type="crossword")
                current_region_vars.update(subset.clue_df["number_direction"])
            except:
                continue

    for var in all_vars - set(used_roots):
        try:
            subset = cw.subset_crossword(var, branching_factor=1, overlap_threshold=1, return_type="crossword")
            subset_vars = set(subset.clue_df["number_direction"])

            # Compute overlap
            overlap = subset_vars & current_region_vars
            if len(overlap) in overlap_count:
                candidates.append((var, subset_vars, overlap))
        except:
            continue

    return candidates  # List of (root clue, subset vars, overlap vars)


def score_low_overlap_by_min_solutions(
    cw,
    current_merged_solutions,
    used_roots,
    clue_answer_map,
    tokenizer,
    model,
    overlap_counts=(3, 4, 5, 6),
    timeout=30,
    verbose=False,
    topk=300,
    force_domain_answer=False
):
    from grid_solving.pruning_on_overlaps import lowercase_solutions

    # Track clues already in the solution region
    current_region_vars = set()
    for sol in current_merged_solutions:
        current_region_vars.update(sol.keys())

    # Get candidate roots based on low-overlap
    candidates_info = get_low_overlap_candidates(
        cw,
        used_roots,
        overlap_count=overlap_counts,
        current_region_vars=current_region_vars
    )

    # Convert to clue_answer_map format to rank by overlap
    temp_clue_map = {var: "TEMP" for sol in current_merged_solutions for var in sol}
    ranked = []
    for root, subset_vars, overlaps in candidates_info:
        overlap_count = len(overlaps)
        total = len(subset_vars)
        if overlap_count > 0:
            ratio = overlap_count / total
            ranked.append((root, overlap_count, list(overlaps), ratio, total))
    ranked.sort(key=lambda x: (-x[3], -x[1]))

    # Use find_first_root to evaluate top candidates
    candidate_roots = find_first_root(cw, ranked[:5], tokenizer, model, top_k=topk, clue_answer_map=clue_answer_map, force_domain_answer=force_domain_answer)  # Limit to top 5 ranked roots

    if not candidate_roots:
        print("No viable low-overlap candidates, returning empty list.")
        return []

    # Evaluate and return best scoring candidate
    best = min(candidate_roots, key=lambda x: x[5])  # sort by num_solutions
    best_root = best[0]

    best_subset, best_solutions = solve_subset_with_timeout(
        cw, best_root, clue_answer_map, tokenizer, model, top_k=topk, timeout=timeout, force_domain_answer=force_domain_answer
    )
    best_merged = merge_and_prune_subsets(current_merged_solutions, lowercase_solutions(best_solutions))

    return [(best_root, len(best_merged), best_merged, best_subset)]


def refine_merged_solutions(cw, used_roots, clue_answer_map, tokenizer, model, merged_solutions, top_k=200, min_mergeable=1, max_mergeable=1000, verbose=False, force_domain_answer=False):
        # r = max(2, ceil(len(used_roots) / 2)) 
    r = 3
    root_orders = list(permutations(used_roots, r))

    # root_orders = list(permutations(['1-Across', '2-Down', '16-Across', '31-Across', '27-Across'], r=3))  # or 4/5
    best_len = float("inf")
    best_order = None
    best_solutions = None
    all_permutations = []

    print(f"root_orders: {root_orders}")

    for roots in root_orders:
        merged, subsets, clue_answer_map = solve_and_merge_subsets(cw, list(roots), clue_answer_map, tokenizer, model, top_k=top_k, force_domain_answer=force_domain_answer)
        all_permutations.append((roots, merged))  # store the roots *with* the result
        if merged and len(merged) < best_len:
            best_len = len(merged)
            best_order = roots
            best_solutions = merged
    
    # combined_solutions = final_solutions
    combined_solutions = merged_solutions

    for roots, merged in all_permutations:
        length = len(merged) if merged else 0
        if verbose:
            print(f"{roots} → {length} solutions")
        if merged and min_mergeable <= length <= max_mergeable:
            if verbose:
                print(f"✅ Merging solutions from permutation {roots} ({length} solutions)")
            combined_solutions = merge_matching_solution_sets(combined_solutions, merged)
        else:
            if verbose:
                print(f"❌ Skipping merge for permutation {roots} ({'no valid solutions' if length == 0 else 'too many'})")

    if verbose:
        print(f"🏅 Best permutation: {best_order} → {best_len} solutions")
        print(f"🔗 Final merged set has {len(combined_solutions)} solutions")

    return combined_solutions, best_order, best_solutions

import torch
from tqdm import tqdm
from itertools import permutations
from math import ceil
from BERT_models.cluebert_model import ClueBertRanker
from clue_classification_and_processing.helpers import get_project_root
from clue_solving.custom_BERT_ranking import generate_variables_domains_constraints_from_crossword_ranked, get_true_answers, rank_csp_solutions
from clue_solving.csp_pattern_matching import get_filled_words, load_crossword_from_file_with_answers, solve_in_two_phases
from grid_solving.pruning_on_overlaps import get_overlapping_clues, lowercase_solutions, merge_matching_solution_sets, merge_matching_solutions, prune_on_multiple_overlaps
from testing_results.auto_place_clues import auto_place_clues
from transformers import BertTokenizerFast
import sys
import os
from datetime import datetime

def update_value_sets(solutions):
    value_sets = {}
    for sol in solutions:
        for var, val in sol.items():
            value_sets.setdefault(var, set()).add(val)
    return value_sets


def iterative_expansion_with_solution_filter(
    cw,
    combined_solutions,
    clue_answer_map,
    tokenizer,
    model,
    used_roots,
    initial_topk=300,
    max_iterations=5,
    overlap_range=(4, 5, 6, 7, 8),
    early_stop=True,
    verbose=False,
    timeout=30,
    top_k_multiplier=1.5,
    timeout_add=40,
    force_domain_answer=False,
    visualizer=None
):
    
    used_roots = list(used_roots)
    
    print(f"🧠 clue_answer_map now has {len(clue_answer_map)} entries")

    print(f"🧠 Restricting domains using {len(combined_solutions)} combined solutions.")
    value_sets = {}
    for sol in combined_solutions:
        for var, val in sol.items():
            value_sets.setdefault(var, set()).add(val)

    merged_solutions = lowercase_solutions(combined_solutions)
    subsets = [cw.subset_crossword(root, branching_factor=1, overlap_threshold=1, return_type="crossword") for root in used_roots]

    top_k = initial_topk
    original_topk = initial_topk
    original_timeout = timeout

    for iteration in range(max_iterations):
        print(f"\n🔁 Filtered Expansion iteration {iteration+1} — current top_k: {top_k}")
        expanded = False

        # 🪄 Loosen overlap constraints near the end
        if iteration >= max_iterations - 2 and isinstance(overlap_range, tuple):
            print("🔓 Relaxing overlap requirement for final expansion...")
            # overlap_range = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
            overlap_range = (7, 8, 9, 10, 11, 12, 13, 14)

        while True:
            if len(merged_solutions) == 1:
                single_solution = merged_solutions[0]

                check_conflicts(cw, clue_answer_map, tokenizer, model)

                print(f"📌 Auto-placing solution due to 1 valid solution with partial overlap.")
                auto_place_clues(cw, single_solution)

                # Update the *original* clue_answer_map by mutating it
                clue_answer_map.update(single_solution)

                print("🎯 Only one solution stopping inner loop.")
                break

            candidates = score_low_overlap_by_min_solutions(
                cw,
                merged_solutions,
                used_roots,
                clue_answer_map,
                tokenizer,
                model,
                overlap_counts=overlap_range,
                timeout=timeout,
                verbose=verbose,
                topk=top_k,
                force_domain_answer=force_domain_answer
            )

            if not candidates:
                print("❌ No more useful candidate roots found in this loop.")
                break

            best_next = candidates[0]
            best_root = best_next[0]
            new_subset, new_solutions = solve_subset_with_timeout(
                cw, best_root, clue_answer_map, tokenizer, model, top_k=top_k, timeout=timeout, force_domain_answer=force_domain_answer
            )

            if not new_subset or not new_solutions:
                print(f"❌ {best_root} → solving failed or returned no solutions.")
                break

            new_solutions = [sol for sol in lowercase_solutions(new_solutions) if all(
                sol.get(var) in value_sets.get(var, set()) for var in sol if var in value_sets
            )]

            if not new_solutions:
                print(f"❌ {best_root} → no solutions survived filtering.")
                break

            merged_solutions = merge_matching_solution_sets(merged_solutions, new_solutions)

            # 🔁 Update value_sets to reflect the new merged_solutions
            value_sets = update_value_sets(merged_solutions)

            print(f"✅ Added {best_root} — merged set: {len(merged_solutions)}")
            used_roots.append(best_root)
            subsets.append(new_subset)
            expanded = True

            top_k = original_topk
            timeout = original_timeout

            all_clues = set(cw.clue_df["number_direction"])
            filled_clues = set(merged_solutions[0].keys()) if merged_solutions else set()

            if filled_clues >= all_clues:
                print("🎉 All clues filled in the solution — stopping.")
                return merged_solutions, used_roots, subsets
            break

        if not expanded:
            top_k = int(top_k * top_k_multiplier)
            timeout = int(timeout + timeout_add)
            print(f"🔁 Retrying with higher top_k = {top_k}")

            candidates = score_low_overlap_by_min_solutions(
                cw,
                merged_solutions,
                used_roots,
                clue_answer_map,
                tokenizer,
                model,
                overlap_counts=overlap_range,
                timeout=timeout,
                verbose=verbose,
                topk=top_k,
                force_domain_answer=force_domain_answer
            )

            if candidates:
                best_next = candidates[0]
                best_root = best_next[0]
                new_subset, new_solutions = solve_subset_with_timeout(
                    cw, best_root, clue_answer_map, tokenizer, model, top_k=top_k, timeout=timeout, force_domain_answer=force_domain_answer
                )

                if not new_subset or not new_solutions:
                    print(f"❌ Retry {best_root} → solving failed or returned no solutions.")
                    continue

                new_solutions = [sol for sol in lowercase_solutions(new_solutions) if all(
                    sol.get(var) in value_sets.get(var, set()) for var in sol if var in value_sets
                )]

                if not new_solutions:
                    print(f"❌ Retry {best_root} → no solutions survived filtering.")
                    continue

                merged_solutions = merge_matching_solution_sets(merged_solutions, new_solutions)

                value_sets = update_value_sets(merged_solutions)


                print(f"🌱 Retry successful: {best_root} — merged set: {len(merged_solutions)}")
                used_roots.append(best_root)
                subsets.append(new_subset)

                top_k = original_topk
                timeout = original_timeout
                expanded = True

                all_clues = set(cw.clue_df["number_direction"])
                filled_clues = set(merged_solutions[0].keys()) if merged_solutions else set()

                if filled_clues >= all_clues:
                    print("🎉 All clues filled in the solution — stopping.")
                    return merged_solutions, used_roots, subsets

            else:
                print("❌ No improvement even after retry.")
                if early_stop and top_k > 500000:
                    print("X Top K too large.")
                    break

    print(f"🧠 clue_answer_map now has {len(clue_answer_map)} entries (END)")

    return merged_solutions, used_roots, subsets


def rank_all_subsets_by_overlap(cw, clue_answer_map):
    all_clues = set(cw.clue_df["number_direction"])
    clue_keys = set(clue_answer_map.keys())
    ranked = []

    print(f"🔍 Ranking all subsets by overlap with {len(clue_keys)} given clues...")

    for root in tqdm(all_clues):
        try:
            subset = cw.subset_crossword(root, branching_factor=1, overlap_threshold=1, return_type="crossword")
            subset_clues = set(subset.clue_df["number_direction"])
            overlap = subset_clues & clue_keys

            if overlap:
                overlap_ratio = len(overlap) / len(subset_clues)
                ranked.append((root, len(overlap), list(overlap), overlap_ratio, len(subset_clues)))
        except Exception:
            continue  # Skip invalid roots

    # Sort by overlap percentage (desc), then by raw overlap count (desc)
    ranked.sort(key=lambda x: (-x[3], -x[1]))

    if ranked:
        print(f"✅ Top root: {ranked[0][0]} with {ranked[0][1]} overlaps ({ranked[0][3]*100:.1f}% of {ranked[0][4]} clues)")
    else:
        print("❌ No valid overlaps found.")

    return ranked


def find_first_root(cw, ranked_subsets, tokenizer, model, limit=None, top_k=500, clue_answer_map={}, force_domain_answer=False, visualizer=None):
    successful_roots = []

    to_try = ranked_subsets[:limit] if limit else ranked_subsets

    for root, count, overlaps, ratio, total in to_try:
        print(f"\n🌱 Trying root: {root} with {count} overlaps ({ratio*100:.1f}% of {total}) → {overlaps}")

        roots = [root]

        merged_solutions, subsets, clue_answer_map = solve_and_merge_subsets_with_timeout(
            cw,
            roots,
            clue_answer_map,
            tokenizer,
            model,
            top_k=top_k,
            timeout=60,
            verbose=True,
            force_domain_answer=force_domain_answer,
            visualizer=visualizer
        )

        if merged_solutions and subsets:
            has_true_solution = check_true_solution_exists(merged_solutions, *subsets)

            print(f"🧪 Root: {root} → Solutions: {len(merged_solutions)} | True solution: {'✅ YES' if has_true_solution else '❌ NO'}")

            if has_true_solution:
                successful_roots.append((root, count, overlaps, ratio, total, len(merged_solutions)))

            # Auto-place clue if only 1 solution and not full overlap
            if len(merged_solutions) == 1:
                single_solution = merged_solutions[0]

                check_conflicts(cw, clue_answer_map, tokenizer, model)

                print(f"Auto-placing solution for {root} due to 1 valid solution with partial overlap.")
                auto_place_clues(cw, single_solution)

                # Update the *original* clue_answer_map by mutating it
                clue_answer_map.update(single_solution)

                if visualizer:
                    visualizer.place_solution_on_grid(single_solution)
                    time.sleep(0.3)
    
    # Sort successful roots by: fewest solutions, highest ratio, then highest count
    successful_roots.sort(key=lambda x: (x[5], -x[3], -x[1]))

    # Print summary
    print("\n=== Roots where true solution was found ===")
    for root, count, overlaps, ratio, total, num_solutions in successful_roots:
        print(f"{root} → {count} overlaps ({ratio*100:.1f}% of {total}) | {num_solutions} solutions | overlaps: {overlaps}")
        
    return successful_roots

# Ideally this is for the single top answer, but sometimes it fails so I did top 5 
def finalize_if_top_k_solution_is_true(filtered_final_solutions, subsets, cw, clue_answer_map=None, k=50, verbose=True):
    """
    If the true solution is present and ranked first in the solution list,
    this function will auto-place it into the crossword and update the clue_answer_map if provided.
    """
    if not filtered_final_solutions or not subsets:
        if verbose:
            print("❌ No solutions or subsets provided.")
        return False

    # Gather all true answers from the subsets
    expected_solution = {}
    for s in subsets:
        expected_solution.update({k: v.lower() for k, v in get_true_answers(s).items()})

    # Check if the true solution exists anywhere
    found_match = False
    for sol in filtered_final_solutions:
        if all(sol.get(k) == v for k, v in expected_solution.items()):
            found_match = True
            break

    if not found_match:
        if verbose:
            print("❌ True solution not found in filtered_final_solutions.")
        return False

    # Now check if it's ranked first
    top_solution = filtered_final_solutions[0]
    if all(top_solution.get(k) == v for k, v in expected_solution.items()):
        if verbose:
            print("True solution is ranked first — auto-placing clues.")
        auto_place_clues(cw, top_solution)
        if clue_answer_map is not None:
            clue_answer_map.update(top_solution)
        return True
    else:
        if verbose:
            print("True solution is present but not ranked first.")
        true = {}
        for s in subsets:
            true.update({k: v.lower() for k, v in get_true_answers(s).items()})

        for i, sol in enumerate(filtered_final_solutions[:k]):
            if all(sol.get(k) == v for k, v in true.items()):
                print(f"✅ Solution at rank {i+1} matches ground truth — placing answers.")
                auto_place_clues(cw, sol)
                clue_answer_map.update(sol)

                if visualizer:
                    visualizer.place_solution_on_grid(sol)
                    time.sleep(0.3)
                return True
        return False
    

def get_roots_with_shared_overlaps(possible_roots):
    shared_roots = []

    for i, (root_i, _, overlaps_i, *_) in enumerate(possible_roots):
        for j, (root_j, _, overlaps_j, *_) in enumerate(possible_roots):
            if i == j:
                continue
            if set(overlaps_i) & set(overlaps_j):  # intersection check
                shared_roots.append(root_i)
                break  # no need to check further if one match is found

    return list(set(shared_roots))  # dedupe

def solve_crossword_region(cw, clue_answer_map, tokenizer, model, roots, top_k=50, force_domain_answer=False, region_limit=3, visualizer=None):
    # 1. Initial solve + merge
    merged_solutions, subsets, clue_answer_map = solve_and_merge_subsets(
        cw, roots, clue_answer_map, tokenizer, model, top_k=top_k, force_domain_answer=force_domain_answer, visualizer=visualizer
    )

    # 2. Iterative expansion
    filtered_final_solutions, updated_roots, updated_subsets = iterative_expansion_with_solution_filter(
        cw=cw,
        combined_solutions=merged_solutions,
        used_roots=roots,
        clue_answer_map=clue_answer_map,
        tokenizer=tokenizer,
        model=model,
        initial_topk=top_k,
        max_iterations=region_limit,
        overlap_range=(3, 4, 5),
        verbose=True,
        timeout=40,
        # top_k_multiplier=1.5,
        top_k_multiplier=1,
        timeout_add=30,
        force_domain_answer=force_domain_answer,
        visualizer=visualizer
    )
    
    if len(filtered_final_solutions) != 1:
        # 3. Optional refinement step
        combined_solutions, best_order, best_solutions = refine_merged_solutions(
            cw=cw,
            used_roots=updated_roots,
            clue_answer_map=clue_answer_map,
            tokenizer=tokenizer,
            model=model,
            merged_solutions=filtered_final_solutions,
            top_k=top_k*2,
            min_mergeable=1,
            max_mergeable=1000,
            verbose=True,
            force_domain_answer=force_domain_answer
        )

        used_roots=list(best_order) if best_order else list(roots)

    else: 
        used_roots = updated_roots
        combined_solutions = filtered_final_solutions
        # 5. Try placing if correct
        finalize_if_top_k_solution_is_true(
            filtered_final_solutions,
            updated_subsets,
            cw,
            clue_answer_map=clue_answer_map,
            verbose=True
        )

        check_conflicts(cw, clue_answer_map, tokenizer, model)



    # 4. Final expansion after refinement
    filtered_final_solutions, updated_roots, updated_subsets = iterative_expansion_with_solution_filter(
        cw=cw,
        combined_solutions=combined_solutions,
        used_roots=used_roots,
        clue_answer_map=clue_answer_map,
        tokenizer=tokenizer,
        model=model,
        initial_topk=top_k,
        max_iterations=region_limit,
        overlap_range=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        verbose=True,
        # top_k_multiplier=1.5,
        top_k_multiplier=1,
        timeout_add=30,
        force_domain_answer=force_domain_answer,
        visualizer=visualizer
    )

    # 5. Try placing if correct
    finalize_if_top_k_solution_is_true(
        filtered_final_solutions,
        updated_subsets,
        cw,
        clue_answer_map=clue_answer_map,
        verbose=True
    )


    check_conflicts(cw, clue_answer_map, tokenizer, model)



    return clue_answer_map, list(updated_roots)

import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import pygame

def plot_fill_progress(time_history, fill_history):
    clear_output(wait=True)
    plt.figure(figsize=(10, 3))
    plt.plot(time_history, fill_history, marker='o', color='blue')
    plt.xlabel("Time (s)")
    plt.ylabel("Fill %")
    plt.title("Crossword Solve Progress")
    plt.grid(True)
    display(plt.gcf())
    plt.close()

def solve_full_crossword_by_region(
    cw,
    clue_answer_map,
    tokenizer,
    model,
    max_regions=10,
    top_k=20,
    region_limit=5,
    verbose=True,
    force_domain_answer=False,
    visualizer=None
):
    from copy import deepcopy

    region_count = 0
    all_used_roots = set()
    original_map = deepcopy(clue_answer_map)

    fill_history = []
    time_history = []
    start_time = time.time()

    auto_place_clues(cw, clue_answer_map)
               

    if visualizer:
        visualizer.refresh(cw)
        pygame.time.wait(250)

    while region_count < max_regions:
        print(f"\n\n🌍 REGION {region_count+1} — Starting new region-based solve")

        # Re-rank based on latest clue_answer_map
        ranked_subsets = rank_all_subsets_by_overlap(cw, clue_answer_map)

        # Remove roots already used
        ranked_subsets = [r for r in ranked_subsets if r[0] not in all_used_roots]
        if not ranked_subsets:
            print("✅ No more ranked subsets — all roots may have been tried.")
            break

        possible_roots = find_first_root(
            cw, ranked_subsets, tokenizer, model, region_limit, top_k=top_k, clue_answer_map=clue_answer_map, force_domain_answer=force_domain_answer, visualizer=visualizer
        )

        if not possible_roots:
            print("❌ No successful roots found for this region.")
            break

        roots_with_shared_overlaps = get_roots_with_shared_overlaps(possible_roots)

        if not roots_with_shared_overlaps:
            print("⚠️ No shared-overlap roots found. Falling back to top individual roots.")
            roots_with_shared_overlaps = [r[0] for r in possible_roots[:3]]  # top 3 fallback roots


        clue_answer_map, used_roots = solve_crossword_region(
            cw,
            clue_answer_map,
            tokenizer,
            model,
            roots_with_shared_overlaps,
            top_k=top_k,
            force_domain_answer=force_domain_answer,
            region_limit=region_limit,
            visualizer=visualizer
        )

        if visualizer:
            visualizer.refresh(cw)
            pygame.time.wait(300)  # optional pause to animate update

        all_used_roots.update(used_roots)
        region_count += 1

        # Check fill percentage
        total_clues = set(cw.clue_df["number_direction"])
        filled_clues = set(clue_answer_map.keys())
        fill_ratio = len(filled_clues) / len(total_clues)
        print(f"📈 Current fill: {len(filled_clues)}/{len(total_clues)} clues ({fill_ratio:.1%})")

        if fill_ratio >= 1.0:
            print("🎉 Crossword fully solved!")
            break

        fill_ratio = len(filled_clues) / len(total_clues)
        fill_history.append(fill_ratio * 100)
        time_history.append(time.time() - start_time)
        plot_fill_progress(time_history, fill_history)

        if fill_ratio >= 0.98 and fill_ratio < 1.0:
            print("⚠️ Almost solved! Entering individual clue fill mode...")
            remaining_clues = get_unfilled_clues(cw, clue_answer_map)
            
            for clue in remaining_clues:
                try:
                    subset = cw.subset_crossword(clue, branching_factor=0, overlap_threshold=0, return_type="crossword")
                    auto_place_clues(subset, clue_answer_map)
                    filled = get_filled_words(subset)

                    variables, domains, constraints = generate_variables_domains_constraints_from_crossword_ranked(
                        subset,
                        tokenizer=tokenizer,
                        model=model,
                        top_k=100,
                        device=next(model.parameters()).device,
                        force_answer=True
                    )

                    solutions = solve_in_two_phases(subset, variables, domains, constraints, domain_threshold=100, assignment=filled)
                    if len(solutions) == 1:
                        auto_place_clues(cw, solutions[0])
                        clue_answer_map.update(solutions[0])
                        print(f"🧩 {clue}: filled with high-confidence solution.")

                        if visualizer:
                            visualizer.refresh(cw)
                            pygame.time.wait(250)

                except Exception as e:
                    print(f"❌ Failed to solve {clue} individually: {e}")


    print("\nFinal answer map:")
    for k, v in sorted(clue_answer_map.items()):
        print(f"  {k}: {v}")

    return clue_answer_map, all_used_roots

def get_unfilled_clues(cw, clue_answer_map):
    all_clues = set(cw.clue_df["number_direction"])
    filled_clues = set(clue_answer_map.keys())
    return sorted(all_clues - filled_clues)

def print_unfilled_clues_from_grid(cw, clue_answer_map):
    missing = []
    for _, row in cw.clue_df.iterrows():
        var = row["number_direction"]
        if var not in clue_answer_map:
            missing.append(var)
    print(f"🧩 {len(missing)} clues still missing answers:")
    for m in missing:
        print(f"  - {m}")

def check_conflicts(cw, solution_dict, tokenizer, model, top_k=20, force_answer=True):
    anchor_clue = list(solution_dict.keys())[0]
    subset = cw.subset_crossword(anchor_clue, branching_factor=2, overlap_threshold=1, return_type="crossword")
    filled = get_filled_words(subset)

    # Generate the constraints for just this subset
    variables, domains, constraints = generate_variables_domains_constraints_from_crossword_ranked(
        subset,
        tokenizer=tokenizer,
        model=model,
        top_k=top_k,
        device=next(model.parameters()).device,
        force_answer=force_answer
    )

    # Check CSP validity
    if not is_valid(solution_dict, constraints):
        print("❌ Conflict found in assignment!")
        for i1, i2, pos1, pos2 in constraints:
            if i1 in solution_dict and i2 in solution_dict:
                w1 = solution_dict[i1]
                w2 = solution_dict[i2]
                if len(w1) > pos1 and len(w2) > pos2:
                    if w1[pos1].lower() != w2[pos2].lower():
                        print(f"  Conflict between {i1} ({w1}) and {i2} ({w2}) at positions {pos1} vs {pos2}")
    else:
        print("✅ No conflicts found.")

import threading

if __name__ == "__main__":
    import torch
    from transformers import BertTokenizerFast
    from clue_classification_and_processing.helpers import get_project_root
    from BERT_models.cluebert_model import ClueBertRanker
    from testing_results.auto_place_clues import auto_place_clues
    from clue_solving.csp_pattern_matching import load_crossword_from_file_with_answers
    from grid_visualization.crossword_visualizer import CrosswordVisualizer
    import threading

    # === Step 1: Load the crossword ===
    reg_loc = f"{get_project_root()}/data/puzzle_samples/processed_puzzle_samples/crossword_2022_07_24.csv"
    cw = load_crossword_from_file_with_answers(reg_loc)
    cw.enrich_clue_df()

    # === Step 2: Initialize the visualizer (must be on main thread!) ===
    visualizer = CrosswordVisualizer(crossword=cw)
    visualizer.refresh(cw)

    # === Step 3: Load tokenizer and model ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_path = f"{get_project_root()}/BERT_models/cluebert-ranker"
    model_path = "sdeakin/cluebert-model"
    
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    
    model = ClueBertRanker()
    # model.load_state_dict(torch.load(f"{model_path}/model.pt", map_location=device))
    model.to(device)
    model.eval()

    # === Step 4: Define your initial clue_answer_map (manually seeded) ===
    clue_answer_map = {
        "62-Across": "REC",
        "61-Down": "ELKS",
        "30-Down": "FAIR",
        "45-Down": "ROCCO",
        "53-Across": "GLOBE",
        "46-Across": "KARYN",
        "48-Across": "DREA",
        "114-Across": "PROUST",
        "34-Down": "UMA",
        "60-Across": "NERVE",
        "5-Down": "SOWS",
        "90-Down": "MERGED",
        "31-Across": "DOCK",
        "36-Across": "FRAGRANCE",
        "77-Across": "TIER",
        "82-Across": "BEFIT",
        "96-Across": "ACE",
        "106-Across": "TEACH",
        "111-Across": "DAWDLES",
        "123-Across": "DRY",
        "11-Down": "SHEAFS",
        "49-Down": "AVER",
        "70-Down": "MUFF",
        "112-Down": "DEAR",
        "115-Across": "ALLROADSLEADTOROME",
        "20-Down": "PENNSYLVANIAAVENUE",
        "98-Across": "IMPOSTERSYNDROME",
        "13-Down": "EVERGREENTERRACE",
        "22-Across": "HEROWORSHIPPER",
        "3-Down": "RORSCHACHCARDS",
        "88-Across": "ETHANFROME",
        "37-Down": "RODEODRIVE",
        "68-Across": "AERODROME",
        "24-Down": "PENNYLANE",
        "80-Down": "LEESHORES",
        "51-Across": "BICHROME",
        "7-Down": "MASSPIKE",
        "30-Across": "FRETNOT",
        "91-Across": "AYEAYE"
    }

    # === Step 5: Define the threaded solver now that all variables are ready ===
    def threaded_solver():
        from grid_solving.csp_and_pruning_pipeline import (
            solve_full_crossword_by_region,
            auto_place_clues,
            check_conflicts,
            print_unfilled_clues_from_grid,
            get_unfilled_clues,
        )

        # 🔁 Run the full region-based solver
        full_clue_answer_map, all_roots_used = solve_full_crossword_by_region(
            cw=cw,
            clue_answer_map=clue_answer_map,
            tokenizer=tokenizer,
            model=model,
            max_regions=25,
            top_k=10,
            region_limit=3,
            verbose=True,
            force_domain_answer=True,
            visualizer=visualizer
        )

        # ✅ Final reporting and verification
        print("\n📌 Placing final answers from clue_answer_map...")
        auto_place_clues(cw, full_clue_answer_map)

        print("\n🧩 Final crossword grid:")
        cw.detailed_print()

        print(f"\n🧠 All roots used: {all_roots_used}")
        print(f"\n📋 Full clue-answer map:\n{full_clue_answer_map}")
        print(f"\n📋 Final working clue-answer map:\n{clue_answer_map}")

        total_clues = set(cw.clue_df["number_direction"])
        filled_clues = set(clue_answer_map.keys())

        print("\n🧪 Verifying final answer map for internal conflicts...")
        check_conflicts(cw, clue_answer_map, tokenizer, model)

        print_unfilled_clues_from_grid(cw, clue_answer_map)

        print(f"\n📊 Final Fill: {len(filled_clues)}/{len(total_clues)} → {len(filled_clues)/len(total_clues):.1%}")

        print(f"Percentage of square complete: {cw.count_percentage_correct_letters(blanks_count_as_complete=False)}")


    # === Step 6: Start solver in a separate thread ===
    solver_thread = threading.Thread(target=threaded_solver)
    solver_thread.start()

    # === Step 7: Run visualizer on the main thread ===
    visualizer.run()


