import pandas as pd

from clue_classification_and_processing.best_match_wiki_page import get_project_root


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
        if raise_error:
            raise ValueError("Cannot validate_clue_df with no path_to_file or df")
        else:
            return "validate_clue_df issue: Cannot validate_clue_df with no path_to_file or df"

    # Raise error if both path_to_file and df are not None
    if path_to_file is not None and df is not None:
        if raise_error:
            raise ValueError("Provide either path_to_file or df, not both.")
        else:
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
                if raise_error:
                    raise ValueError("File must be a .csv or .xlsx")
                else:
                    return "validate_clue_df issue: File must be a .csv or .xlsx"

    # Raise error for missing columns or extra columns
    required_columns = ["number", "start_col", "start_row", "end_col", "end_row", "clue"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    # Missing
    if missing_columns:
        if raise_error:
            raise ValueError(f"Missing required columns: {missing_columns}")
        else:
            return f"validate_clue_df issue: Missing required columns: {missing_columns}"


    # Confirm that "number", "start_col", "start_row", "end_col", "end_row" are all ints
    int_columns = ["number", "start_col", "start_row", "end_col", "end_row"]
    non_int_columns = []
    for col in int_columns:
        if not pd.api.types.is_numeric_dtype(df[col]) or not (df[col].dropna() % 1 == 0).all():
            non_int_columns.append(col)
    if non_int_columns:
        if raise_error:
            raise ValueError(f"Expected integer columns: {non_int_columns}")
        else:
            return f"validate_clue_df issue: Expected integer columns: {non_int_columns}"

    # Check for more than 2 instances of "number" (only can have across and down, no more than that!)
    counts = df["number"].value_counts()
    too_many = counts[counts > 2].index.tolist()
    if too_many:
        if raise_error:
            raise ValueError(f'"number" values appear more than twice: {too_many}')
        else:
            return f'validate_clue_df issue: "number" values appear more than twice: {too_many}'

    # if any clues are blank, raise error
    if df["clue"].isna().any() or (df["clue"].astype(str).str.strip() == "").any():
        if raise_error:
            raise ValueError('Some entries in "clue" are blank or missing.')
        else:
            return 'validate_clue_df issue: Some entries in "clue" are blank or missing.'

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
        return abs(clue_entry["end_col"] - clue_entry["start_col"]) + abs(clue_entry["end_row"] - clue_entry["start_row"]) + 1

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

    def enrich_clue_df(self):
        # For each entry in self.clue_df, if start_col == end_col then set
        # self.clue_df["number_direction"] to self.clue_df["number"] +
        # "-" + "down" (same logic for start_row == end_row)
        # Also set clue length
        self.clue_df["number_direction"] = self.clue_df.apply(self.get_direction, axis=1)
        self.clue_df["length"] = self.clue_df.apply(self.get_length, axis=1)

        # If optional column is included, use it to check lengths and then delete that optional column
        self.optional_check_length()


big_loc = f"{get_project_root()}/data/puzzle_samples/wednesday_03262025.xlsx"
mini_loc = f"{get_project_root()}/data/puzzle_samples/mini_03262025.xlsx"
df = pd.read_excel(big_loc)
mini_crossword = Crossword(clue_df=df)