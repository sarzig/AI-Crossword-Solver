import random
import numpy as np
import pandas as pd
from nltk.corpus import words
from clue_classification_and_processing.helpers import conditional_raise, print_if, get_project_root, \
    get_clues_dataframe
from web.nyt_html_to_standard_csv import get_random_clue_df
import sys
import os
import atexit

from clue_classification_and_processing.helpers import conditional_raise, print_if, get_project_root
from clue_classification_and_processing.helpers import get_project_root


# xxx sheryl replace with actual function call. Right now leaving as global
nltk_vocab = [x.lower() for x in words.words()]
big_clue_df = get_clues_dataframe()
crossword_vocab = set(big_clue_df["Word"].dropna().astype(str).str.lower())
VOCAB = set(nltk_vocab).union(crossword_vocab)


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

    # If df is provided but not a DataFrame, raise TypeError
    if not isinstance(df, pd.DataFrame):
        conditional_raise(TypeError("Provided df is not a pandas DataFrame."), raise_error)
        return "validate_clue_df issue: df is not a pandas DataFrame."

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
        self.number_fillable_spaces = np.sum(self.grid != "■")

        # If solving using subsets, this tracks the subsets.
        # subset list is a dict of Crossword objects where the key is something like "5-Across"
        # or "42-Down"
        self.subset_dict = {}

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

        optional_col = "answer (optional column, for checking only)"
        if optional_col in self.clue_df.columns:
            if self.clue_df[optional_col].notna().any():
                answer_lengths = self.clue_df[optional_col].dropna().astype(str).str.len()
                expected_lengths = self.clue_df.loc[answer_lengths.index, "length"]
                mismatched = self.clue_df.loc[answer_lengths.index][answer_lengths != expected_lengths]
                if not mismatched.empty:
                    details = mismatched[["number", optional_col, "length"]].to_dict(orient="records")
                    raise ValueError(f"Answer length mismatch detected in optional column: {details}")
            #self.clue_df.drop(columns=[optional_col], inplace=True) # don't want to do this anymore!

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

        # Add coordinate set - allows for faster sub-setting
        self.clue_df["coordinate_set"] = self.clue_df.apply(
            lambda row: self.get_coordinate_set(
                start_col=row["start_col"],
                start_row=row["start_row"],
                end_col=row["end_col"],
                end_row=row["end_row"]
            ),
            axis=1
        )

    def get_coordinate_set(self, start_col, start_row, end_col, end_row):
        iterate_row = end_row > start_row

        # For "Down" clues, iterate the row downwards
        if iterate_row:
            coordinates = [(start_col, y) for y in range(start_row, end_row + 1)]
        else:
            coordinates = [(x, start_row) for x in range(start_col, end_col + 1)]

        return coordinates

    def place_helper_answers(self, fill_oov=True, fill_percent=None):
        """
        This method is to aid in troubleshooting.

        It places clues on an empty grid.

        It can do the following:
            * Place random clues up to n percentage of the board
            * Fill all OOV clues in

        If fill_oov is True AND fill_percent/fill_number is non-None, it will first
        place all OOVs and THEN if the percent filled is less than fill_percent,
        it will randomly place more.

        :return: nothing, just updates puzzle
        """

        # make sure there's nothing in puzzle
        if self.calculate_completion_percentage_by_char() > 0:
            print("place_helper_clues is meant for an empty crossword.")
            return

        # make sure there's an answers column in self.clue_df
        if "answer (optional column, for checking only)" not in self.clue_df.columns:
            print("no answer column - can't place helper starter words.")

        # Fill out of vocabulary words
        if fill_oov:
            for index, row in self.clue_df.iterrows():
                answer = row["answer (optional column, for checking only)"]

                # Fill in all words in the vocab
                if answer.lower() not in VOCAB:
                    grid_loc = row["number_direction"]
                    self.place_word(word=answer,
                                    grid_location=grid_loc)

        all_indices = list(self.clue_df.index)

        if fill_percent is not None:
            # If the percent_complete is not none, keep filling random words until
            # percent complete is equal to or higher than fill_percent
            while self.calculate_completion_percentage_by_char() < fill_percent:
                index = random.choice(all_indices)
                row = self.clue_df.loc[index]
                answer = row.get("answer (optional column, for checking only)")
                number_direction = row["number_direction"]

                self.place_word(answer, number_direction,
                                allow_overwriting=True,
                                flag_errors=False)

    def add_coordinate_set(self):
        """
        Go through dataframe
        :return:
        """

    def generate_grid(self):
        # Create a blank crossword grid filled with "■"
        grid = np.full((self.grid_height, self.grid_width), "■", dtype=object)

        # Based on contents of the clue_df, remove the [X]s:
        for _, clue in self.clue_df.iterrows():
            start_row = clue["start_row"]
            start_col = clue["start_col"]
            end_row = clue["end_row"]
            end_col = clue["end_col"]

            # Determine direction
            if start_row == end_row:  # Across
                for c in range(start_col, end_col + 1):
                    grid[start_row][c] = " "
            elif start_col == end_col:  # Down
                for r in range(start_row, end_row + 1):
                    grid[r][start_col] = " "
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
            line = "|   " + "".join(f"[{cell}]" for cell in row) + "   |"
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
        empties = (self.grid == " ").sum()
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

        ### xxx tbd to do

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
            target = f"{char}"
            if "Across" in grid_location:
                current = self.grid[start_row][start_col + i]
                if current != " " and current != target:
                    if not allow_overwriting:
                        raise ValueError(f"Cannot overwrite non-empty cell at ({start_row}, {start_col + i})")
                    if flag_overwriting:
                        print(f"Overwriting cell ({start_row}, {start_col + i}) from {current} to {target}")
                self.grid[start_row][start_col + i] = target

            elif "Down" in grid_location:
                current = self.grid[start_row + i][start_col]
                if current != " " and current != target:
                    if not allow_overwriting:
                        raise ValueError(f"Cannot overwrite non-empty cell at ({start_row + i}, {start_col})")
                    if flag_overwriting:
                        print(f"Overwriting cell ({start_row + i}, {start_col}) from {current} to {target}")
                self.grid[start_row + i][start_col] = target

            else:
                raise ValueError(f"Unsupported direction in location: {grid_location}")

    def subset_crossword(self, grid_location, branching_factor=1, threshold=0.25, return_type="add"):
        """
        Create a subset of the crossword that includes all clues that intersect
        with the given clue (via a shared coordinate). If branching_factor is 1, it
        will get all words that intersect with the original word.

        If branching_factor is 2, it will get all words that intersect with the
        result of branching factor of 1.

        Once the branches have been set, there is a threshhold for which other
        clues are included in the subset. If a clue is a 100% subset of clues
        previously covered, it will always be subsetted. Otherwise, clues are
        added to the subset based on threshold - a 4-letter clue with 2 letters
        overlapping with subsetted clues will be included for a threshold of 0.5 or
        above.

        :param branching_factor: How many degres away from first word should be subsetted
        :param threshold: percent threshold after which secondary clues should be considered part of subset
        as well
        :param grid_location: location like "21-Across"
        :returns: nothing, but adds to self.subset_dict with the grid_location as key
        """

        if grid_location not in self.clue_df["number_direction"].values:
            raise ValueError(f"Clue {grid_location} not found in the clue dataframe.")

        # Get the coordinate set of the target clue - this is initial set of coordinates
        next_pass_coordinates = self.clue_df.set_index("number_direction").at[grid_location, "coordinate_set"]

        # Get the subset of the clue_df for which one or more coordinates is in initial_word_coordinates
        intersections = {grid_location: "original word"}
        for i in range(branching_factor + 1):
            if len(intersections) == len(self.clue_df):
                break

            coordinates_to_match = list(set(next_pass_coordinates))
            next_pass_coordinates = []

            # Iterate across dataframe (yes, it's inefficient, but these are 100 rows only)
            for idx, row in self.clue_df.iterrows():

                # If the word is already captured in intersections, skip
                if row["number_direction"] in intersections.keys():
                    continue

                # Final pass: apply threshold rule
                if i == branching_factor:
                    overlap = set(row["coordinate_set"]) & set(coordinates_to_match)
                    percent_overlap = len(overlap) / len(row["coordinate_set"])

                    if percent_overlap >= threshold or set(row["coordinate_set"]).issubset(coordinates_to_match):
                        intersections[row["number_direction"]] = f"{i} pass match: {round(percent_overlap * 100)}%"
                        next_pass_coordinates += row["coordinate_set"]
                        continue

                # Earlier passes: any coordinate match is enough
                else:
                    if any(coord in coordinates_to_match for coord in row["coordinate_set"]):
                        intersections[row["number_direction"]] = f"{i} pass intersection"
                        next_pass_coordinates += row["coordinate_set"]
                        continue

        if return_type == "dict":
            return intersections
        elif return_type == "df":
            return self.clue_df[self.clue_df["number_direction"].isin(intersections.keys())].copy()
        elif return_type == "crossword":
            return Crossword(self.clue_df[self.clue_df["number_direction"].isin(intersections.keys())].copy())
        elif return_type == "add to subset list":
            self.subset_dict[grid_location] = Crossword(self.clue_df[self.clue_df["number_direction"].isin(intersections.keys())].copy())

'''
df = get_random_clue_df(return_type="random regular")
my_crossword = Crossword(clue_df=df)
my_crossword.detailed_print()
my_crossword.place_helper_answers(fill_oov=True)
my_crossword.detailed_print()
'''

lg_loc = f"{get_project_root()}/data/puzzle_samples/processed_puzzle_samples/crossword_2024_06_12.csv"

'''
lg_loc = f"{get_project_root()}/data/puzzle_samples/processed_puzzle_samples/crossword_2024_06_12.csv"
med_loc = f"{get_project_root()}/data/puzzle_samples/wednesday_03262025.xlsx"
mini_loc = f"{get_project_root()}/data/puzzle_samples/mini_03262025.xlsx"

df = pd.read_csv(lg_loc)
my_crossword = Crossword(clue_df=df)
my_crossword.detailed_print()
#result = my_crossword.place_word("hello", "5-across")

intersection_dict = my_crossword.subset_crossword("1-Across", 1, threshold=2/3, return_type="add to subset list")
#full_subset = pd.concat([subset, additional])
#subset_crossword = Crossword(subset)
#subset_crossword.detailed_print()

#for key in intersection_dict.keys():
#    print(f"{key}: {intersection_dict[key]}")
'''

def get_random_clue_df(puzzle_type="any", return_filename=False, force_previous=True):
    """
    Loads a random clue DataFrame from processed_puzzle_samples.

    Args:
        puzzle_type: "mini" (default), "standard", or "any"
            - "mini": allows puzzles up to 9x9
            - "standard": allows puzzles up to 15x15
            - "any": allows all puzzle sizes
        return_filename: if True, returns a tuple (df, filename)
        force_previous: if True, will reuse last_loaded_crossword.txt if it exists

    Returns:
        df, filename (if return_filename=True)
        or
        df (if return_filename=False)
    """
    root = get_project_root()
    sample_dir = os.path.join(root, "data", "puzzle_samples", "processed_puzzle_samples")
    last_used_file = os.path.join(root, "last_loaded_crossword.txt")

    # Set max size filter
    if puzzle_type == "mini":
        max_size = 10
    elif puzzle_type == "standard":
        max_size = 16
    elif puzzle_type == "any":
        max_size = float("inf")
    else:
        raise ValueError("Invalid puzzle_type. Use 'mini', 'standard', or 'any'.")

    # Use previous file if available
    if force_previous and os.path.exists(last_used_file):
        with open(last_used_file, "r") as f:
            chosen_file = f.read().strip()
        full_path = os.path.join(sample_dir, chosen_file)
        if os.path.exists(full_path):
            df = pd.read_csv(full_path)
            print(f"[INFO] Reusing previously loaded crossword file: {chosen_file}")
            return (df, chosen_file) if return_filename else df

    # Pick new file
    csv_files = [f for f in os.listdir(sample_dir) if f.endswith(".csv")]
    random.shuffle(csv_files)

    for chosen_file in csv_files:
        df = pd.read_csv(os.path.join(sample_dir, chosen_file))
        if df["end_row"].max() < max_size and df["end_col"].max() < max_size:
            with open(last_used_file, "w") as f:
                f.write(chosen_file)
            print(f"[INFO] Selected and saved crossword file: {chosen_file}")
            return (df, chosen_file) if return_filename else df

    raise ValueError(f"No suitable crossword found for puzzle_type: {puzzle_type}")

def get_saved_or_new_crossword(force_new=False):

    """
    Main function to retrieve a crossword.

    - If force_new=True, selects and caches a new crossword puzzle CSV.
    - If force_new=False, reuses the one stored in 'last_loaded_crossword.txt'.

    Returns:
        crossword: Crossword object built from the selected puzzle
        filename: Filename of the CSV loaded
    """
     
    
    df, filename = get_random_clue_df(return_filename=True, force_previous=not force_new)
    
    crossword = Crossword(clue_df=df, table_name=filename)
    
    # ✅ Automatically print the crossword grid
    print(f"\n[INFO] Loaded clue file: {filename}")
    crossword.detailed_print()

    return crossword, filename

def delete_last_loaded_crossword_on_exit():
    last_file = os.path.join(get_project_root(), "last_loaded_crossword.txt")
    if os.path.exists(last_file):
        os.remove(last_file)
        print("[INFO] Deleted last_loaded_crossword.txt on program exit.")

atexit.register(delete_last_loaded_crossword_on_exit)
