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


class CustomCrosswordVisualizer(CrosswordVisualizer):
    def __init__(self, crossword):
        print("[DEBUG] Initializing CustomCrosswordVisualizer...")
        super().__init__(crossword)

        self.highlight_cells = set()
        self.highlight_words = set()
        self.clue_text = ""

        # 🔁 Resize window to make space for clue text
        
        # Calculate extra height for clues
        extra_height = 20 * len(crossword.clue_df)  # Space for clues
        self.WINDOW_HEIGHT = self.GRID_HEIGHT * (self.CELL_SIZE + self.MARGIN) + self.MARGIN + 310

        # Calculate a more optimal window width based on the longest clue
        # Find the clue that needs the most width
        max_clue_length = max([len(f"{num}-{dir}. {clue}") 
                              for num, dir, clue in zip(
                                  crossword.clue_df["number"],
                                  crossword.clue_df["direction"],
                                  crossword.clue_df["clue"]
                              )], default=0)
        
        # Estimate characters per line - approx 6-7px per character at font size 16
        # Adding a small buffer of 20px for padding, but keeping it compact
        min_width_for_clues = int(max_clue_length * 6.5) + 20
        
        # Determine grid width
        grid_width = self.GRID_WIDTH * (self.CELL_SIZE + self.MARGIN) + self.MARGIN
        
        # Set window width to the larger of the two, but cap it at a reasonable size
        # to avoid excessive empty space
        reasonable_max_width = 650  # Reasonable width for readability without too much empty space
        self.WINDOW_WIDTH = min(reasonable_max_width, max(grid_width, min_width_for_clues))

        # Resize window with new dimensions
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        
        print("[DEBUG] SCREEN HEIGHT:", self.WINDOW_HEIGHT)
        print("[DEBUG] SCREEN WIDTH:", self.WINDOW_WIDTH)


    def refresh(self, crossword, highlight_cells=None, highlight_words=None, clue_text=""):
        self.crossword = crossword
        self.highlight_cells = highlight_cells or set()
        self.highlight_words = highlight_words or set()
        self.clue_text = clue_text
        self._prepare_crossword_elements()

    def wrap_text(self, text, font, max_width):
        """Break text into lines that fit within max_width"""
        words = text.split(' ')
        lines = []
        current_line = words[0]
        
        for word in words[1:]:
            test_line = current_line + ' ' + word
            text_width = font.size(test_line)[0]
            if text_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        
        lines.append(current_line)  # Add the last line
        return lines

    def draw_grid(self):
        self.screen.fill(self.GRAY)
        
        # Calculate grid centering adjustments for a better layout
        grid_pixel_width = self.GRID_WIDTH * (self.CELL_SIZE + self.MARGIN) + self.MARGIN
        x_offset = (self.WINDOW_WIDTH - grid_pixel_width) // 2  # Center the grid horizontally
        
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                x = c * (self.CELL_SIZE + self.MARGIN) + self.MARGIN + x_offset
                y = r * (self.CELL_SIZE + self.MARGIN) + self.MARGIN
                rect = pygame.Rect(x, y, self.CELL_SIZE, self.CELL_SIZE)

                cell_value = self.crossword.grid[r][c]
                is_black = cell_value == "■"

                # Background color
                if is_black:
                    color = self.BLACK
                elif (r, c) in self.highlight_cells:
                    color = self.YELLOW
                elif (r, c) in self.clue_numbers:
                    color = self.LIGHT_GRAY
                else:
                    color = self.WHITE

                pygame.draw.rect(self.screen, color, rect)

                # Highlight word borders
                if (r, c) in self.highlight_words:
                    pygame.draw.rect(self.screen, (255, 80, 80), rect, 3)
                else:
                    pygame.draw.rect(self.screen, self.BLACK, rect, 1)

                # Clue number
                if (r, c) in self.clue_numbers:
                    num_surface = self.number_font.render(str(self.clue_numbers[(r, c)]), True, self.BLACK)
                    self.screen.blit(num_surface, (x + 2, y + 2))

                # Letter
                if cell_value and not is_black:
                    letter_surface = self.letter_font.render(cell_value.upper(), True, self.BLACK)
                    text_rect = letter_surface.get_rect(center=(x + self.CELL_SIZE // 2, y + self.CELL_SIZE // 2 + 3))
                    self.screen.blit(letter_surface, text_rect)

        # Render clue text below the grid with wrapping
        if hasattr(self.crossword, "clue_df"):
            clue_font = pygame.font.SysFont('Arial', 16)
            clues = self.crossword.clue_df["clue"].tolist()
            numbers = self.crossword.clue_df["number"].tolist()
            directions = self.crossword.clue_df["direction"].tolist()
            
            y_pos = self.GRID_HEIGHT * (self.CELL_SIZE + self.MARGIN) + 10
            text_padding = 20
            max_width = self.WINDOW_WIDTH - (text_padding * 2)  # Padding on each side
            
            # Draw a title for the clues section
            title_font = pygame.font.SysFont('Arial', 18, bold=True)
            title_surface = title_font.render("Clues", True, (0, 0, 0))
            self.screen.blit(title_surface, (text_padding, y_pos))
            y_pos += 25  # Slightly more space after title
            
            # Determine if we should display clues in two columns
            halfway = len(clues) // 2
            use_two_columns = self.WINDOW_WIDTH >= 500 and len(clues) > 6
            
            for i in range(len(clues)):
                # Skip to second column if needed
                if use_two_columns and i == halfway:
                    y_pos = self.GRID_HEIGHT * (self.CELL_SIZE + self.MARGIN) + 35  # Reset Y position
                    text_padding = self.WINDOW_WIDTH // 2 + 10  # Start second column
                
                # Format the clue with number and direction
                full_clue = f"{numbers[i]}-{directions[i]}. {clues[i]}"
                
                # Apply text wrapping
                wrapped_lines = self.wrap_text(full_clue, clue_font, max_width // (2 if use_two_columns else 1))
                
                for line in wrapped_lines:
                    clue_surface = clue_font.render(line, True, (0, 0, 0))
                    self.screen.blit(clue_surface, (text_padding, y_pos))
                    y_pos += 20  # Line height

        pygame.display.flip()


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

    def get_clue_text(self, clue_label):
        """Returns the clue string from the clue_df given a label like '1-Across'."""
        try:
            number, direction = clue_label.split("-")
            number = int(number)
            row = self.clue_df[(self.clue_df["number"] == number) & (self.clue_df["direction"] == direction)]
            if not row.empty:
                return row.iloc[0]["clue"]
        except Exception as e:
            print(f"[ERROR] Couldn't find clue for {clue_label}: {e}")
        return ""


    def get_blank_grid(self):
        """Creates a blank grid with None (black squares) in all positions initially."""
        max_row = max(self.clue_df["start_row"].max(), self.clue_df["end_row"].max()) + 1
        max_col = max(self.clue_df["start_col"].max(), self.clue_df["end_col"].max()) + 1
        return [["■" for _ in range(max_col)] for _ in range(max_row)]
    
    def get_word_cells(self, clue_label):
        """
        Given a clue label like '3-Down', return all (row, col) cell coordinates it fills.
        """
        clue_row = self.clue_df[
            (self.clue_df["number"] == int(clue_label.split("-")[0])) &
            (self.clue_df["direction"] == clue_label.split("-")[1])
        ]

        if clue_row.empty:
            return []

        row = int(clue_row["start_row"].values[0])
        col = int(clue_row["start_col"].values[0])
        end_row = int(clue_row["end_row"].values[0])
        end_col = int(clue_row["end_col"].values[0])

        cells = []
        if row == end_row:  # Horizontal (Across)
            for c in range(col, end_col + 1):
                cells.append((row, c))
        elif col == end_col:  # Vertical (Down)
            for r in range(row, end_row + 1):
                cells.append((r, col))

        return cells


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
        """Displays crossword solutions with visual highlights for letter changes and word borders."""
        self.visualizer = CustomCrosswordVisualizer(Crossword(clue_df=self.clue_df))
        prev_grid = None
        manual_mode = False  # Toggle to False for autoplay
        first_run = True

        running = True
        while running and self.current_index < self.total:
            crossword, score = self.load_crossword()

            changed_cells = self.get_changed_cells(prev_grid, crossword.grid) if not first_run else set()

            current_clue = ""

            # Highlight full word if any letter in it changes
            highlight_words = set()
            if changed_cells:
                for clue_label, word in self.ranked_solutions[self.current_index][0].items():
                    for (r, c) in self.get_word_cells(clue_label):
                        if (r, c) in changed_cells:
                            highlight_words.update(self.get_word_cells(clue_label))
                            current_clue = self.get_clue_text(clue_label)
                            break

            self.visualizer.refresh(
                crossword,
                highlight_cells=changed_cells if not first_run else set(),
                highlight_words=highlight_words if not first_run else set(),
                clue_text=current_clue if not first_run else ""
            )

            print(f"Solution {self.current_index + 1}/{self.total} | Score: {score}")
            self.visualizer.draw_grid()
            pygame.display.flip()

            # Store the current grid before moving to the next one
            prev_grid = [row.copy() for row in crossword.grid]
            first_run = False

            # === Manual vs Auto navigation ===
            wait_start = time.time()
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            pygame.quit()
                            return
                        elif event.key == pygame.K_RIGHT:
                            self.current_index = (self.current_index + 1) % self.total
                            break  # Break inner while, re-loop outer while
                        elif event.key == pygame.K_LEFT:
                            self.current_index = (self.current_index - 1) % self.total
                            break
                else:
                    # If not in manual mode, advance after 0.5s
                    if not manual_mode and time.time() - wait_start > 0.7:
                        self.current_index += 1
                        break
                    time.sleep(0.02)
                    continue
                break  # Break inner while only on keypress

        pygame.quit()


# === Usage example ===
if __name__ == "__main__":
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

    # Sample ranked solutions (replace with your actual model predictions)
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

    # Start the solution navigator
    navigator = SolutionNavigator(ranked_solutions, clue_df)
    navigator.start()