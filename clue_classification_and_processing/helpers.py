import os
import pandas as pd

"""
This is the generic helpers file. 

Generally functions will be put here if they work well, do not have a better "home" 
elsewhere, and are polished.

Author: Sarah Witzig

"""


def print_if(statement, print_bool):
    """
    Helper to simplify conditional printing.

    :param statement: statement to print
    :param print_bool: boolean to print or not
    """
    if print_bool:
        print(statement)


def conditional_raise(error, raise_bool):
    """
    Helper to simplify the conditional raising of errors.

    :param error: error to raise
    :param raise_bool: boolean to raise
    """

    if raise_bool:
        raise error


def get_clues_dataframe():
    """
    Uses OS lib to search for cwd, and then walks back to project root.

    :return: the main Kaggle dataframe with all clues
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
