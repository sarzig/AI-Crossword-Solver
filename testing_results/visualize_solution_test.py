import pandas as pd
import os
import sys
import pygame
import time

# Add project root to sys.path to enable imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import Crossword data structure and visualizer
from puzzle_objects.crossword_and_clue import Crossword
from grid_visualization.crossword_visualizer import CrosswordVisualizer


# === Class to allow interactive solution navigation using Pygame ===
class SolutionNavigator:
    def __init__(self, ranked_solutions, clue_df):
        """
        Args:
            ranked_solutions: list of tuples (solution_dict, score)
                Each solution_dict maps "clue_label" (e.g., '1-Across') to the guessed word.
            clue_df: DataFrame containing crossword clues and positions
        """
        self.ranked_solutions = ranked_solutions
        self.clue_df = clue_df
        self.current_index = 0  # Index of currently viewed solution
        self.total = len(ranked_solutions)
        self.visualizer = None

        pygame.init()
        pygame.display.set_caption("Crossword Solution Navigator")

    def get_blank_grid(self):
        """Creates a blank grid with None (black squares) in all positions initially."""
        max_row = max(self.clue_df["start_row"].max(), self.clue_df["end_row"].max()) + 1
        max_col = max(self.clue_df["start_col"].max(), self.clue_df["end_col"].max()) + 1
        return [["■" for _ in range(max_col)] for _ in range(max_row)]

    def load_crossword(self):
        """
        Loads the current solution into a Crossword object with a blank grid,
        and places the words based on clue labels. Converts None to "" to avoid visualizer crash.
        """
        crossword = Crossword(clue_df=self.clue_df)
        crossword.grid = self.get_blank_grid()

        solution_dict, score = self.ranked_solutions[self.current_index]
        for clue_label, word in solution_dict.items():
            crossword.place_word(word, clue_label, allow_overwriting=True, flag_errors=False)

        # Patch: convert None → "" so visualizer won't break
        for r in range(len(crossword.grid)):
            for c in range(len(crossword.grid[0])):
                if crossword.grid[r][c] is None:
                    crossword.grid[r][c] = ""

        return crossword, score
    

    @staticmethod
    def get_changed_cells(prev_grid, new_grid):
        changed = set()
        if prev_grid is None:
            return changed  # Nothing to compare on the first run
        for r in range(len(new_grid)):
            for c in range(len(new_grid[0])):
                prev = prev_grid[r][c]
                curr = new_grid[r][c]
                if prev != curr:
                    changed.add((r, c))
        return changed

    def start(self):
        """Automatically cycles through each ranked solution every 0.5s, colors changing letters, exits after last."""
        self.visualizer = CrosswordVisualizer(Crossword(clue_df=self.clue_df))
        prev_grid = None  # Track previous grid

        while self.current_index < self.total:
            crossword, score = self.load_crossword()

            # Detect and highlight changed cells
            changed_cells =  self.get_changed_cells(prev_grid, crossword.grid)
            for r, c in changed_cells:
                if crossword.grid[r][c] is not None:
                    crossword.grid[r][c] = crossword.grid[r][c].lower()


            self.visualizer.refresh(crossword)
            print(f"Solution {self.current_index+1}/{self.total} | Score: {score}")
            self.visualizer.draw_grid()
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    return

            prev_grid = [row.copy() for row in crossword.grid]  # Save for next comparison
            self.current_index += 1
            time.sleep(0.5)

        time.sleep(1)
        pygame.quit()


# === Load CSV and Preprocess Clue Data ===

# Load the CSV file containing the crossword clues
csv_path = os.path.join("data", "puzzle_samples", "processed_puzzle_samples", "mini_2024_03_02.csv")
clue_df = pd.read_csv(csv_path, index_col=0)



# Reset index so clue number becomes a regular column
clue_df = clue_df.reset_index(names="number")

# Add a simple 'direction' column for clue orientation
# This assumes first half of clues are Across, second half are Down
halfway = len(clue_df) // 2
clue_df["direction"] = ["Across"] * halfway + ["Down"] * (len(clue_df) - halfway)

# Drop optional columns that cause validation errors during Crossword object construction
for optional_col in [
    "answer (optional column, for checking only)",
    "length (optional column, for checking only)"
]:
    if optional_col in clue_df.columns:
        clue_df.drop(columns=[optional_col], inplace=True)

# Final safety check

# === Precomputed Ranked Solutions ===
# Each item is a tuple: (solution_dict, confidence_score)
# Replace this list with actual predictions from your model
# Sample best solution dictionary from your data
ranked_solutions = [({'1-Across': 'MATH', '10-Across': 'RBG', '12-Across': 'SIMMER', '1-Down': 'MAMA', '4-Down': 'HMM', '6-Down': 'NORMAL', '7-Down': 'INBETA', '11-Down': 'GREW', '5-Across': 'alumni', '8-Across': 'mormon', '15-Across': 'claw', '2-Down': 'alonso', '14-Across': 'opiate', '13-Down': 'mic', '3-Down': 'turnip', '9-Across': 'ann'}, 9.244984205812216), 
                 ({'1-Across': 'MATH', '10-Across': 'RBG', '12-Across': 'SIMMER', '1-Down': 'MAMA', '4-Down': 'HMM', '6-Down': 'NORMAL', '7-Down': 'INBETA', '11-Down': 'GREW', '5-Across': 'alumni', '8-Across': 'mormon', '15-Across': 'alaw', '2-Down': 'aloose', '14-Across': 'estate', '13-Down': 'mta', '3-Down': 'turkis', '9-Across': 'aok'}, 9.12752802297473), 
                 ({'1-Across': 'MATH', '10-Across': 'RBG', '12-Across': 'SIMMER', '1-Down': 'MAMA', '4-Down': 'HMM', '6-Down': 'NORMAL', '7-Down': 'INBETA', '11-Down': 'GREW', '5-Across': 'alumni', '8-Across': 'mormon', '15-Across': 'alaw', '2-Down': 'alonso', '14-Across': 'opiate', '13-Down': 'mia', '3-Down': 'turnip', '9-Across': 'ann'}, 8.953922387212515), 
                 ({'1-Across': 'MATH', '10-Across': 'RBG', '12-Across': 'SIMMER', '1-Down': 'MAMA', '4-Down': 'HMM', '6-Down': 'NORMAL', '7-Down': 'INBETA', '11-Down': 'GREW', '5-Across': 'alumni', '8-Across': 'mormon', '15-Across': 'alaw', '2-Down': 'alonso', '14-Across': 'optate', '13-Down': 'mta', '3-Down': 'turnip', '9-Across': 'ann'}, 8.911350432783365), 
                 ({'1-Across': 'MATH', '10-Across': 'RBG', '12-Across': 'SIMMER', '1-Down': 'MAMA', '4-Down': 'HMM', '6-Down': 'NORMAL', '7-Down': 'INBETA', '11-Down': 'GREW', '5-Across': 'alumni', '8-Across': 'mormon', '15-Across': 'slaw', '2-Down': 'aloose', '14-Across': 'estate', '13-Down': 'mts', '3-Down': 'turkis', '9-Across': 'aok'}, 8.720243785530329), 
                 ({'1-Across': 'MATH', '10-Across': 'RBG', '12-Across': 'SIMMER', '1-Down': 'MAMA', '4-Down': 'HMM', '6-Down': 'NORMAL', '7-Down': 'INBETA', '11-Down': 'GREW', '5-Across': 'alumni', '8-Across': 'mormon', '15-Across': 'blaw', '2-Down': 'alonso', '14-Across': 'onlate', '13-Down': 'mlb', '3-Down': 'turnin', '9-Across': 'ann'}, 8.694568064063787), 
                 ({'1-Across': 'MATH', '10-Across': 'RBG', '12-Across': 'SIMMER', '1-Down': 'MAMA', '4-Down': 'HMM', '6-Down': 'NORMAL', '7-Down': 'INBETA', '11-Down': 'GREW', '5-Across': 'alumni', '8-Across': 'mormon', '15-Across': 'alaw', '2-Down': 'alonso', '14-Across': 'oxeate', '13-Down': 'mea', '3-Down': 'turnix', '9-Across': 'ann'}, 8.656534042209387), 
                 ({'1-Across': 'MATH', '10-Across': 'RBG', '12-Across': 'SIMMER', '1-Down': 'MAMA', '4-Down': 'HMM', '6-Down': 'NORMAL', '7-Down': 'INBETA', '11-Down': 'GREW', '5-Across': 'alumni', '8-Across': 'mormon', '15-Across': 'blaw', '2-Down': 'alonso', '14-Across': 'opiate', '13-Down': 'mib', '3-Down': 'turnip', '9-Across': 'ann'}, 8.632688697427511), 
                 ({'1-Across': 'MATH', '10-Across': 'RBG', '12-Across': 'SIMMER', '1-Down': 'MAMA', '4-Down': 'HMM', '6-Down': 'NORMAL', '7-Down': 'INBETA', '11-Down': 'GREW', '5-Across': 'alumni', '8-Across': 'mormon', '15-Across': 'slaw', '2-Down': 'alonso', '14-Across': 'onlate', '13-Down': 'mls', '3-Down': 'turnin', '9-Across': 'ann'}, 8.59223161265254), 
                 ({'1-Across': 'MATH', '10-Across': 'RBG', '12-Across': 'SIMMER', '1-Down': 'MAMA', '4-Down': 'HMM', '6-Down': 'NORMAL', '7-Down': 'INBETA', '11-Down': 'GREW', '5-Across': 'alumni', '8-Across': 'mormon', '15-Across': 'slaw', '2-Down': 'alonso', '14-Across': 'optate', '13-Down': 'mts', '3-Down': 'turnip', '9-Across': 'ann'}, 8.504066195338964), 
                 ({'1-Across': 'MATH', '10-Across': 'RBG', '12-Across': 'SIMMER', '1-Down': 'MAMA', '4-Down': 'HMM', '6-Down': 'NORMAL', '7-Down': 'INBETA', '11-Down': 'GREW', '5-Across': 'alumni', '8-Across': 'mormon', '15-Across': 'slaw', '2-Down': 'aloose', '14-Across': 'ensate', '13-Down': 'mss', '3-Down': 'turnin', '9-Across': 'aon'}, 8.42264261469245), 
                 ({'1-Across': 'MATH', '10-Across': 'RBG', '12-Across': 'SIMMER', '1-Down': 'MAMA', '4-Down': 'HMM', '6-Down': 'NORMAL', '7-Down': 'INBETA', '11-Down': 'GREW', '5-Across': 'alumni', '8-Across': 'mormon', '15-Across': 'alaw', '2-Down': 'aloose', '14-Across': 'exmate', '13-Down': 'mma', '3-Down': 'turnix', '9-Across': 'aon'}, 8.415652554482222), 
                 ({'1-Across': 'MATH', '10-Across': 'RBG', '12-Across': 'SIMMER', '1-Down': 'MAMA', '4-Down': 'HMM', '6-Down': 'NORMAL', '7-Down': 'INBETA', '11-Down': 'GREW', '5-Across': 'alumni', '8-Across': 'mormon', '15-Across': 'slaw', '2-Down': 'alonso', '14-Across': 'opiate', '13-Down': 'mis', '3-Down': 'turnip', '9-Across': 'ann'}, 8.37682762334589), 
                 ({'1-Across': 'MATH', '10-Across': 'RBG', '12-Across': 'SIMMER', '1-Down': 'MAMA', '4-Down': 'HMM', '6-Down': 'NORMAL', '7-Down': 'INBETA', '11-Down': 'GREW', '5-Across': 'alumni', '8-Across': 'mormon', '15-Across': 'slaw', '2-Down': 'alonso', '14-Across': 'oxeate', '13-Down': 'mes', '3-Down': 'turnix', '9-Across': 'ann'}, 8.226721700280905), 
                 ({'1-Across': 'MATH', '10-Across': 'RBG', '12-Across': 'SIMMER', '1-Down': 'MAMA', '4-Down': 'HMM', '6-Down': 'NORMAL', '7-Down': 'INBETA', '11-Down': 'GREW', '5-Across': 'alumni', '8-Across': 'mormon', '15-Across': 'slaw', '2-Down': 'aloose', '14-Across': 'exmate', '13-Down': 'mms', '3-Down': 'turnix', '9-Across': 'aon'}, 7.961682487279177)]



# === END TEST BLOCK ===

if __name__ == "__main__":
    navigator = SolutionNavigator(ranked_solutions, clue_df)
    navigator.start()
