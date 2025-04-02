from letter_pattern_matching import find_words_by_pattern

def is_valid(assignment, constraints):
    for i1, i2, pos1, pos2 in constraints:
        if i1 in assignment and i2 in assignment:
            w1, w2 = assignment[i1], assignment[i2]
            if len(w1) <= pos1 or len(w2) <= pos2:
                return False
            if w1[pos1] != w2[pos2]:
                return False
    return True

def solve_csp(variables, domains, constraints, assignment={}):
    if len(assignment) == len(variables):
        return [assignment.copy()]

    solutions = []
    var = [v for v in variables if v not in assignment][0]

    for value in domains[var]:
        assignment[var] = value
        if is_valid(assignment, constraints):
            solutions.extend(solve_csp(variables, domains, constraints, assignment))
        del assignment[var]

    return solutions

### 
# Partial CSP
###

# def solve_all_max_partial_csp(variables, domains, constraints):
#     def backtrack(assignment, all_solutions, max_len):
#         # Track the longest valid assignments
#         if is_valid(assignment, constraints):
#             current_len = len(assignment)
#             if current_len > max_len[0]:
#                 max_len[0] = current_len
#                 all_solutions.clear()
#                 all_solutions.append(assignment.copy())
#             elif current_len == max_len[0]:
#                 all_solutions.append(assignment.copy())

#         # Explore further assignments
#         unassigned = [v for v in variables if v not in assignment and domains[v]]
#         if not unassigned:
#             return

#         var = min(unassigned, key=lambda v: len(domains[v]))  # heuristic: smallest domain first

#         for value in domains[var]:
#             assignment[var] = value
#             if is_valid(assignment, constraints):
#                 backtrack(assignment, all_solutions, max_len)
#             del assignment[var]

#     all_solutions = []
#     max_len = [0]
#     backtrack({}, all_solutions, max_len)
#     return all_solutions

def solve_all_max_partial_csp(variables, domains, constraints):
    def backtrack(assignment, unassigned, all_solutions, max_len):
        if is_valid(assignment, constraints):
            current_len = len(assignment)
            if current_len > max_len[0]:
                max_len[0] = current_len
                all_solutions.clear()
                all_solutions.append(assignment.copy())
            elif current_len == max_len[0]:
                all_solutions.append(assignment.copy())

        for var in unassigned:
            for value in domains[var]:
                assignment[var] = value
                if is_valid(assignment, constraints):
                    backtrack(
                        assignment,
                        [v for v in unassigned if v != var],
                        all_solutions,
                        max_len,
                    )
                del assignment[var]

    all_solutions = []
    max_len = [0]
    valid_vars = [v for v in variables if domains[v]]
    backtrack({}, valid_vars, all_solutions, max_len)
    return all_solutions

#########################################################################################
# Generate variables, domains, constraints 
#########################################################################################

from collections import defaultdict

def generate_variables_domains_constraints(clues):
    grid_map = defaultdict(list)  # (row, col) → list of (var_id, letter_index)

    variables = []
    domains = {}
    constraints = set()  # use set to avoid duplicates

    for clue in clues:
        var_id = clue["number"]
        row, col = clue["row"], clue["col"]
        length = clue["length"]
        direction = clue["direction"]

        variables.append(var_id)
        # Use pattern if present, otherwise fall back to generic dots
        pattern = clue.get("pattern", "." * length)
        domains[var_id] = find_words_by_pattern(pattern)

        # Map this clue's letters onto the grid
        for i in range(length):
            r = row + i if direction == "down" else row
            c = col + i if direction == "across" else col
            grid_map[(r, c)].append((var_id, i, direction))

    print("=== Grid Map ===")
    for pos, entries in sorted(grid_map.items()):
        print(f"{pos}: {entries}")

    # Generate constraints only between crossing words
    for square, entries in grid_map.items():
        if len(entries) > 1:
            for i in range(len(entries)):
                for j in range(i+1, len(entries)):
                    var1, idx1, dir1 = entries[i]
                    var2, idx2, dir2 = entries[j]
                    if dir1 != dir2:  # Only create constraints if directions differ
                        constraints.add((var1, var2, idx1, idx2))
                        constraints.add((var2, var1, idx2, idx1))  # bidirectional

    return variables, domains, list(constraints)

#########################################################################################
# Testing - clueset 1
#########################################################################################

clues = [
    {"number": 1.1, "row": 0, "col": 0, "direction": "across",   "length": 4, "pattern": "..s."},
    # {"number": 15, "row": 1, "col": 0, "direction": "across",    "length": 4},
    {"number": 18, "row": 2, "col": 0, "direction": "across",    "length": 4, "pattern": "i.t."},
    {"number": 21, "row": 3, "col": 0, "direction": "across",  "length": 5, "pattern": "crime"},
    # {"number": 25, "row": 4, "col": 0, "direction": "across",  "length": 6},
    # {"number": 31, "row": 5, "col": 0, "direction": "across",  "length": 3},

    {"number": 1, "row": 0, "col": 0, "direction": "down", "length": 6, "pattern": "cliche"},
    {"number": 2, "row": 0, "col": 1, "direction": "down", "length": 6},
    # {"number": 3, "row": 0, "col": 2, "direction": "down", "length": 6},
    # {"number": 4, "row": 0, "col": 3, "direction": "down", "length": 5},
]

variables, domains, constraints = generate_variables_domains_constraints(clues)

print(f"variables: {variables}")
# print(f"domains: {domains}")
print(f"constraints: {constraints}")


partial_solutions = solve_all_max_partial_csp(variables, domains, constraints)

if not partial_solutions:
    print("No valid partial solution found.")
else:
    print(f"Found {len(partial_solutions)} partial solution(s) of max length {len(partial_solutions[0])}:\n")
    for i, sol in enumerate(partial_solutions, 1):
        print(f"partial_solutions {i}:")
        for var in sorted(sol):
            print(f"  Clue {var}: {sol[var]}")
    print(f"Number of partial solutions: {len(partial_solutions)}")
    

#########################################################################################
# Testing - clueset 2
#########################################################################################


