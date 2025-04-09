import ast


def extract_solution_sets_from_text(text_block):
    """
    Extract solution sets from a formatted string like:
    'ranked: [({...}, score), ...]'

    Returns:
        List[Set[str]]: each set contains entries like '12-Across: SIMMER'
    """
    # Step 1: Strip "ranked: " and parse the list using ast.literal_eval
    prefix = "ranked:"
    if text_block.startswith(prefix):
        text_block = text_block[len(prefix):].strip()

    parsed = ast.literal_eval(text_block)

    # Step 2: Extract and format each solution dict into a set of "clue: word" strings
    formatted_sets = []
    for solution_dict, score in parsed:
        sol_set = {f"{k}: {v}".upper() for k, v in solution_dict.items()}
        formatted_sets.append(sol_set)

    return formatted_sets

a = extract_solution_sets_from_text()