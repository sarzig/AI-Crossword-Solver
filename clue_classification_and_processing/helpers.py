import os


def print_if(statement, print_bool):
    if print_bool:
        print(statement)


def conditional_raise(error, raise_bool):
    # helper to simplify the conditional raising
    if raise_bool:
        raise error


def get_clues_dataframe():
    """
    Uses OS lib to search for cwd, and then walks back to project root.

    :return:
    """
    # get cwd and split into constituent parts
    cwd = os.getcwd()
    path_parts = cwd.split(os.sep)

    # Look for project name in the path
    root = ""
    if "ai_crossword_solver" in path_parts:
        index = path_parts.index("ai_crossword_solver")
        root = os.sep.join(path_parts[:index + 1])

    # Load dataset
    clues_path = os.path.join(root, r"data//nytcrosswords.csv")

    clues_df = pd.read_csv(clues_path, encoding='latin1')
    return clues_df