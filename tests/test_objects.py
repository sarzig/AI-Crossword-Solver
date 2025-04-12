"""
Author: Sarah

Some basic tests for testing crossword creation.

*Gen AI Assisted*
"""

import pandas as pd
import pytest
from puzzle_objects.crossword_and_clue import validate_clue_df


# 1. No arguments provided
def test_no_arguments():
    with pytest.raises(ValueError, match="Cannot validate_clue_df with no path_to_file or df"):
        validate_clue_df()


# 2. Both df and path_to_file provided
def test_both_arguments():
    dummy_df = pd.DataFrame()
    with pytest.raises(ValueError, match="Provide either path_to_file or df, not both."):
        validate_clue_df(path_to_file="dummy.csv", df=dummy_df)


# 3. Missing required columns
def test_missing_required_columns():
    df = pd.DataFrame(columns=["number", "start_col", "start_row"])  # missing others
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_clue_df(df=df)


# 4. Extra/unexpected columns
def test_unexpected_columns():
    df = pd.DataFrame(columns=[
        "number", "start_col", "start_row", "end_col", "end_row", "clue text", "extra"
    ])
    with pytest.raises(ValueError, match="Unexpected columns found"):
        validate_clue_df(df=df)


# 5. Non-integer numeric values (e.g., 1.5 in int column)
def test_non_integer_numeric_column():
    df = pd.DataFrame({
        "number": [1, 2],
        "start_col": [0.0, 1.0],
        "start_row": [1.5, 2.0],  # Invalid
        "end_col": [2, 3],
        "end_row": [3, 4],
        "clue text": ["First clue", "Second clue"]
    })
    with pytest.raises(ValueError, match="Expected integer columns"):
        validate_clue_df(df=df)


# 6. String in numeric column
def test_string_in_integer_column():
    df = pd.DataFrame({
        "number": ["one", "two"],  # Invalid
        "start_col": [0, 1],
        "start_row": [1, 2],
        "end_col": [2, 3],
        "end_row": [3, 4],
        "clue text": ["Clue one", "Clue two"]
    })
    with pytest.raises(ValueError, match="Expected integer columns"):
        validate_clue_df(df=df)
