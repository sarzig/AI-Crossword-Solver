import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clue_classification_and_processing.helpers import conditional_raise, print_if, get_project_root


def validate_clue_df(path_to_file=None, df=None, raise_error=True):
    """
    Given either a path to csv or a dataframe, validate if the data is
    in the proper format for entry into a Crossword object.

    # genai assisted

    :param raise_error: True if you want to raise an error, otherwise return
                        info about what's wrong
    :param path_to_file: path to the csv file
    :param df: dataframe with columns
                        * "number"
                        * "start_col"
                        * "start_row"
                        * "end_col"
                        * "end_row"
                        * "clue text"

    :return: True if validated, or error text if not
    """

    # Null checks
    if path_to_file is None and df is None:
        conditional_raise(ValueError("Cannot validate_clue_df with no path_to_file or df"), raise_error)
        return "validate_clue_df issue: Cannot validate_clue_df with no path_to_file or df"

    # Raise error if both path_to_file and df are not None
    if path_to_file is not None and df is not None:
        conditional_raise(ValueError("Provide either path_to_file or df, not both."), raise_error)
        return "validate_clue_df issue: Provide either path_to_file or df, not both."

    # Check if the file is xlsx or csv. If neither, raise appropriate error.
    # If it is xlsx or csv, read into a dataframe named df
    if path_to_file is not None:
        if path_to_file is not None:
            if path_to_file.endswith(".csv"):
                df = pd.read_csv(path_to_file)
            elif path_to_file.endswith(".xlsx"):
                df = pd.read_excel(path_to_file)
            else:
                conditional_raise(ValueError("File must be a .csv or .xlsx"), raise_error)
                return "validate_clue_df issue: File must be a .csv or .xlsx"

    # Raise error for missing columns
    required_columns = ["number", "start_col", "start_row", "end_col", "end_row", "clue"]
    # optional_columns = ['length (optional column, for checking only)', 'answer (optional column, for checking only)']
    missing_columns = [col for col in required_columns if col not in df.columns]

    # Missing
    if missing_columns:
        conditional_raise(ValueError(f"Missing required columns: {missing_columns}"), raise_error)
        return f"validate_clue_df issue: Missing required columns: {missing_columns}"

    # Confirm that "number", "start_col", "start_row", "end_col", "end_row" are all ints
    int_columns = ["number", "start_col", "start_row", "end_col", "end_row"]
    non_int_columns = []
    for col in int_columns:
        if not pd.api.types.is_numeric_dtype(df[col]) or not (df[col].dropna() % 1 == 0).all():
            non_int_columns.append(col)
    if non_int_columns:
        conditional_raise(ValueError(f"Expected integer columns: {non_int_columns}"), raise_error)
        return f"validate_clue_df issue: Expected integer columns: {non_int_columns}"

    # Check for more than 2 instances of "number" (only can have across and down, no more than that!)
    counts = df["number"].value_counts()
    too_many = counts[counts > 2].index.tolist()
    if too_many:
        conditional_raise(ValueError(f'"number" values appear more than twice: {too_many}'), raise_error)
        return f'validate_clue_df issue: "number" values appear more than twice: {too_many}'

    # if any clues are blank, raise error
    if df["clue"].isna().any() or (df["clue"].astype(str).str.strip() == "").any():
        conditional_raise(ValueError('Some entries in "clue" are blank or missing.'), raise_error)
        return 'validate_clue_df issue: Some entries in "clue" are blank or missing.'

    # Confirm no duplicates of the combination ["start_col", "start_row", "end_col", "end_row"]
    coordinate_columns = ["start_col", "start_row", "end_col", "end_row"]
    duplicated_coordinates = df.duplicated(subset=coordinate_columns)
    if duplicated_coordinates.any():
        dupe_rows = df[duplicated_coordinates].to_dict(orient="records")
        conditional_raise(ValueError(f"Duplicate coordinate entries found: {dupe_rows}"), raise_error)
        return f"validate_clue_df issue: Duplicate coordinate entries found: {dupe_rows}"

    # Confirm all ["start_col", "start_row", "end_col", "end_row"] are positive
    negative_coordinates = df[coordinate_columns][df[coordinate_columns] < 0].dropna(how="all")
    if not negative_coordinates.empty:
        bad_rows = df[df[coordinate_columns].lt(0).any(axis=1)].to_dict(orient="records")
        conditional_raise(ValueError(f"Negative values found in coordinates: {bad_rows}"), raise_error)
        return f"validate_clue_df issue: Negative values found in coordinates: {bad_rows}"

    # If we've reached this point with no errors, it's validated
    return True


class Crossword:
    def __init__(self, clue_df, table_name=None):

        # Check the source data meets requirements. If so, store clue_df
        validate_result = validate_clue_df(df=clue_df, raise_error=True)
        self.clue_df = clue_df

        # Enrich the clue df by adding "x-across" / "y-down" notation,
        self.enrich_clue_df()

        # Set some grid attributes
        self.grid_height = int(self.clue_df["end_row"].max() + 1)  # max row + 1
        self.grid_width = int(self.clue_df["end_col"].max() + 1)  # max column + 1
        self.table_name = table_name

        # Generate the numpy array to store the solved grid, where spaces with [X] are
        # the black spaces of a crossword
        self.grid = self.generate_grid()

        # Count number of clues and number of white spaces - useful for calculations
        self.number_clues = len(self.clue_df)
        self.number_fillable_spaces = np.sum(self.grid != "[■]")

    @staticmethod
    def get_direction(clue_entry):
        start = f'start=({clue_entry["start_col"]}, {clue_entry["start_row"]})'
        end = f'end=({clue_entry["end_col"]}, {clue_entry["end_row"]})'

        if clue_entry["start_col"] == clue_entry["end_col"] and clue_entry["start_row"] == clue_entry["end_row"]:
            raise ValueError(f"Start and end positions are the same — direction cannot be determined. {start}, {end}")
        elif clue_entry["start_col"] == clue_entry["end_col"]:
            return f'{clue_entry["number"]}-Down'
        elif clue_entry["start_row"] == clue_entry["end_row"]:
            return f'{clue_entry["number"]}-Across'
        else:
            raise ValueError(
                f"Clue is not strictly across or down — diagonal or irregular shape detected. {start}, {end}")

    @staticmethod
    def get_length(clue_entry):
        return abs(clue_entry["end_col"] - clue_entry["start_col"]) + abs(
            clue_entry["end_row"] - clue_entry["start_row"]) + 1

    def optional_check_length(self):
        # If there's a column called 'length (optional column, for checking only)',
        # then check if there are any mismatch between that column and "length".
        # If so, raise error with details about mismatch.
        # If not, delete optional column.
        optional_col = "length (optional column, for checking only)"
        if optional_col in self.clue_df.columns:
            if self.clue_df[optional_col].notna().any():
                mismatched = self.clue_df[self.clue_df[optional_col] != self.clue_df["length"]]
                if not mismatched.empty:
                    details = mismatched[["number", optional_col, "length"]].to_dict(orient="records")
                    raise ValueError(f"Length mismatch detected in optional column: {details}")
            self.clue_df.drop(columns=[optional_col], inplace=True)

    def optional_check_answer_length(self):
        # If there's a column called 'answer (optional column, for checking only)',
        # then check if there are any mismatch between that column's word length and "length".
        # If so, raise error with details about mismatch.
        # If not, delete optional column.
        optional_col = "answer (optional column, for checking only)"
        if optional_col in self.clue_df.columns:
            if self.clue_df[optional_col].notna().any():
                answer_lengths = self.clue_df[optional_col].dropna().astype(str).str.len()
                expected_lengths = self.clue_df.loc[answer_lengths.index, "length"]
                mismatched = self.clue_df.loc[answer_lengths.index][answer_lengths != expected_lengths]
                if not mismatched.empty:
                    details = mismatched[["number", optional_col, "length"]].to_dict(orient="records")
                    raise ValueError(f"Answer length mismatch detected in optional column: {details}")
            self.clue_df.drop(columns=[optional_col], inplace=True)

    def enrich_clue_df(self):
        # For each entry in self.clue_df, if start_col == end_col then set
        # self.clue_df["number_direction"] to self.clue_df["number"] +
        # "-" + "down" (same logic for start_row == end_row)
        # Also set clue length
        self.clue_df["number_direction"] = self.clue_df.apply(self.get_direction, axis=1)
        self.clue_df["length"] = self.clue_df.apply(self.get_length, axis=1)

        # If optional column is included, use it to check lengths and then delete that optional column
        self.optional_check_length()
        self.optional_check_answer_length()

    def generate_grid(self):
        # Create a blank crossword grid filled with "[X]"
        grid = np.full((self.grid_height, self.grid_width), "[■]", dtype=object)

        # Based on contents of the clue_df, remove the [X]s:
        for _, clue in self.clue_df.iterrows():
            start_row = clue["start_row"]
            start_col = clue["start_col"]
            end_row = clue["end_row"]
            end_col = clue["end_col"]

            # Determine direction
            if start_row == end_row:  # Across
                for c in range(start_col, end_col + 1):
                    grid[start_row][c] = "[ ]"
            elif start_col == end_col:  # Down
                for r in range(start_row, end_row + 1):
                    grid[r][start_col] = "[ ]"
            else:
                raise ValueError(f"Invalid clue direction (not across or down): {clue.to_dict()}")

        return grid

    def grid_string(self):
        # Returns string with plain text grid info
        lines = []
        width = self.grid.shape[1]
        horizontal_border = "+---" + "---" * width + "---+"
        empty_line = "|   " + "   " * width + "   |"

        lines.append(horizontal_border)
        lines.append(empty_line)  # bordered blank line after top border
        for row in self.grid:
            line = "|   " + "".join(cell for cell in row) + "   |"
            lines.append(line)
        lines.append(empty_line)  # bordered blank line before bottom border
        lines.append(horizontal_border)

        return "\n".join(lines)

    def detailed_print(self):
        print("+" + "-" * (len("|   " + "".join(self.grid[0]) + "   |") - 2) + "+")
        print("Crossword Info")

        # Title (optional)
        if self.table_name:
            print(f"Title: {self.table_name}")

        # Grid size
        print(f"Grid size: {self.grid_height} rows × {self.grid_width} columns")

        # Percent completion (based on how many [ ] vs non-[X] cells)
        print(f"Fill progress: {self.calculate_completion_percentage_by_char():.0f}%")

        # Grid
        print("\nGrid:\n" + self.grid_string())

    def calculate_completion_percentage_by_char(self):
        # Percent completion (based on how many [ ] vs non-[X] cells)
        empties = (self.grid == "[ ]").sum()
        percent_complete = ((self.number_fillable_spaces - empties) / self.number_fillable_spaces * 100)

        return percent_complete

    def count_number_correct_words(self):
        """
        If self.clue_df has answers column AND it is filled, then count how many words
        are correctly filled out in self.grid. A word is considered correct if it contains
        no blanks in the numpy array, and if, when concatenated from (start_col, start_row) to
        (end_col, end_row), it is correct.

        :return:
        """
        optional_col = "answer (optional column, for checking only)"
        if optional_col not in self.clue_df.columns or self.clue_df[optional_col].isna().all():
            print("No filled answer column found — skipping correct word count.")
            return None

    def count_number_correct_letters(self):
        """
        If self.clue_df has answers column AND it is filled, then count how many LETTERS
        are correctly filled out in self.grid.

        :return:
        """
        optional_col = "answer (optional column, for checking only)"
        if optional_col not in self.clue_df.columns or self.clue_df[optional_col].isna().all():
            print("No filled answer column found — skipping correct letter count.")
            return None

    def place_word(self, word, grid_location,
                   allow_overwriting=True,
                   flag_overwriting=False,
                   raise_errors=False,
                   flag_errors=True):
        """
        Place a word into the grid, given a grid location.

        :param word: word to be placed
        :param grid_location: location like "21-Across"
        :param allow_overwriting: if True, this will overwrite existing characters
        :param flag_overwriting: if True (and if allow_overwriting is True, this will overwrite with a
                                 print statement)
        :param flag_errors:
        :param raise_errors:
        :return:
        """
        # Null check
        if word is None or grid_location is None:
            error_statement = "Need word and valid grid location."
            conditional_raise(ValueError(error_statement), raise_errors)
            print_if(error_statement, flag_errors)
            return False

        # Confirm word only contains AZ chars and uppercase it
        word = word.upper()
        if not word.isalpha():
            error_statement = f"Word '{word}' must only contain letters A-Z."
            conditional_raise(ValueError(error_statement), raise_errors)
            print_if(error_statement, flag_errors)
            return False

        # Confirm grid location exists and if so, harmonize the way to refer to it
        clue = self.clue_df[self.clue_df["number_direction"].str.lower() == grid_location.lower()]
        if clue.empty:
            error_statement = f"Grid location '{grid_location}' not found in clues."
            conditional_raise(ValueError(error_statement), raise_errors)
            print_if(error_statement, flag_errors)
            return False

        grid_location = clue.iloc[0]["number_direction"]  # put grid_location in exact format from self.clue_df

        # Confirm word is correct length
        expected_length = clue.iloc[0]["length"]
        if len(word) != expected_length:
            error_statement = f"Word '{word}' length {len(word)} does not match "\
                              f"expected length {expected_length} for {grid_location}."
            conditional_raise(ValueError(error_statement), raise_errors)
            print_if(error_statement, flag_errors)
            return False

        # Place the word and return True or False
        try:
            self.place_word_onto_numpy_array(word, grid_location, allow_overwriting, flag_overwriting)
            return True
        except Exception as e:
            conditional_raise(e, raise_errors)
            print_if(e, flag_errors)
            return False

    def place_word_onto_numpy_array(self, word, grid_location, allow_overwriting, flag_overwriting):
        # This places a word, assumes that it is called within place_word_wrapper() which
        # does some pre-processing on word and grid_location

        # Get the specific clue entry
        clue = self.clue_df[self.clue_df["number_direction"] == grid_location]  # get row for this grid-location
        clue = clue.iloc[0]  # convert to series
        start_row = clue["start_row"]
        start_col = clue["start_col"]

        for i, char in enumerate(word):
            target = f"[{char}]"
            if "Across" in grid_location:
                current = self.grid[start_row][start_col + i]
                if current != "[ ]" and current != target:
                    if not allow_overwriting:
                        raise ValueError(f"Cannot overwrite non-empty cell at ({start_row}, {start_col + i})")
                    if flag_overwriting:
                        print(f"Overwriting cell ({start_row}, {start_col + i}) from {current} to {target}")
                self.grid[start_row][start_col + i] = target

            elif "Down" in grid_location:
                current = self.grid[start_row + i][start_col]
                if current != "[ ]" and current != target:
                    if not allow_overwriting:
                        raise ValueError(f"Cannot overwrite non-empty cell at ({start_row + i}, {start_col})")
                    if flag_overwriting:
                        print(f"Overwriting cell ({start_row + i}, {start_col}) from {current} to {target}")
                self.grid[start_row + i][start_col] = target

            else:
                raise ValueError(f"Unsupported direction in location: {grid_location}")


lg_loc = f"{get_project_root()}/data/puzzle_samples/processed_puzzle_samples/crossword_2022_06_12.csv"
med_loc = f"{get_project_root()}/data/puzzle_samples/wednesday_03262025.xlsx"
mini_loc = f"{get_project_root()}/data/puzzle_samples/mini_03262025.xlsx"

df = pd.read_csv(lg_loc)
my_crossword = Crossword(clue_df=df)
my_crossword.detailed_print()
result = my_crossword.place_word("hello", "5-across")