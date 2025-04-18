#########################################################################################
# Getting all the minis names from folder
#########################################################################################

# Step 1: Find all mini puzzle files
import os
import re
from clue_classification_and_processing.helpers import get_project_root


puzzle_dir = f"{get_project_root()}/data/puzzle_samples/processed_puzzle_samples"
all_files = os.listdir(puzzle_dir)

mini_dates = []
for f in all_files:
    match = re.match(r"mini_(\d{4}_\d{2}_\d{2})\.(csv|xlsx)", f)
    if match:
        mini_dates.append(match.group(1))

mini_dates.sort()

# Step 2: Print available files
print("Found puzzle dates:")
for date in mini_dates:
    print(f" - {date}")

# Step 3: Create the list to edit
print("\nUse this list to manually comment out slow puzzles:")
print("mini_files_to_run = [")
for date in mini_dates:
    print(f'    "mini_{date}",')
print("]")