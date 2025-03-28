from datetime import datetime
import os
import random
import re
import pandas as pd
from bs4 import BeautifulSoup
from objects.crossword_and_clue import Crossword
from clue_classification_and_processing.helpers import get_project_root


"""
This entire parsing was created with ChatGPT.

Using saved html, this converts a crossword into a meaningful dataframe / saves
to csv.

Functions:
 * get_coordinates(x, y, cell_size, cell_offset)
 * puzzle_html_to_df(filename)
 * get_random_clue_df(folder=r"data/puzzle_samples/raw_html/", return_type="All")
 * process_all_raw_html_to_csv() - looks for all html files in raw_html and converts to csv
 * rename_puzzles() - helper to rename puzzles from NYT download format to my format
"""


def get_coordinates(x, y, cell_size, cell_offset):
    col = round((float(x) - cell_offset) / cell_size)
    row = round((float(y) - cell_offset) / cell_size)
    return row, col


def puzzle_html_to_df(filename):
    """
    Parse a saved crossword HTML file into a DataFrame with clue positions.

    Required columns: number, start_col, start_row, end_col, end_row, clue
    Optional columns: length (optional column, for checking only),
                      answer (optional column, for checking only) — if all letters are present
    """

    with open(filename, 'r', encoding='utf-8') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')

    # Step 1: Extract clues
    clue_list = []
    sections = soup.select('.xwd__clue-list--wrapper')

    for section in sections:
        direction_el = section.select_one('.xwd__clue-list--title')
        direction = direction_el.text.strip().lower() if direction_el else "across"

        for clue_li in section.select('li.xwd__clue--li'):
            number_el = clue_li.select_one('.xwd__clue--label')
            text_el = clue_li.select_one('.xwd__clue--text')
            if number_el and text_el:
                clue_list.append({
                    'number': int(number_el.text.strip()),
                    'clue': text_el.text.strip(),
                    'direction': direction
                })

    # Step 2: Extract and organize cells
    cell_blocks = soup.select('g.xwd__cell')
    grid_map = {}

    # Dynamically detect cell size and offset
    cell_rects = soup.select('g.xwd__cell rect')
    positions = []

    for rect in cell_rects:
        if 'x' in rect.attrs and 'y' in rect.attrs:
            x = float(rect['x'])
            y = float(rect['y'])
            positions.append((x, y))

    positions = sorted(set(positions))
    min_x = min(pos[0] for pos in positions)
    min_y = min(pos[1] for pos in positions)
    cell_offset = min(min_x, min_y)

    first_row = [pos for pos in positions if pos[1] == min_y]
    first_row = sorted(first_row, key=lambda p: p[0])
    if len(first_row) > 1:
        cell_size = first_row[1][0] - first_row[0][0]
    else:
        raise ValueError("Not enough horizontal cells to compute cell size.")

    for cell_block in cell_blocks:
        rect = cell_block.select_one('rect')
        text_tags = cell_block.select('text')
        cell_id_match = re.search(r'cell-id-(\d+)', rect.get('id', ''))
        if not cell_id_match:
            continue

        cell_id = int(cell_id_match.group(1))
        x, y = rect['x'], rect['y']
        row, col = get_coordinates(x, y, cell_size=cell_size, cell_offset=cell_offset)

        clue_number = None
        for t in text_tags:
            text_val = t.get_text(strip=True)
            if text_val.isdigit():
                clue_number = int(text_val)
                break

        is_black = 'xwd__cell--block' in rect['class']
        grid_map[(row, col)] = {
            'cell_id': cell_id,
            'is_black': is_black,
            'clue_number': clue_number
        }

    # Step 3: Map clue numbers to positions
    number_to_position = {}
    for (r, c), cell in grid_map.items():
        if cell['clue_number'] is not None:
            number_to_position[cell['clue_number']] = (r, c)

    # Step 4: Trace clues across/down, calculate bounds and extract answers if fully revealed
    full_output = []
    answers_available = False

    for clue in clue_list:
        number = clue['number']
        direction = clue['direction']
        clue_text = clue['clue']

        if number not in number_to_position:
            full_output.append({
                "number": number,
                "start_col": -1,
                "start_row": -1,
                "end_col": -1,
                "end_row": -1,
                "clue": clue_text,
                "length (optional column, for checking only)": 0
            })
            continue

        start_r, start_c = number_to_position[number]
        r, c = start_r, start_c
        cells_in_clue = []
        answer_letters = []

        while (r, c) in grid_map and not grid_map[(r, c)]['is_black']:
            cells_in_clue.append((r, c))

            cell_id = grid_map[(r, c)]["cell_id"]
            cell_selector = f'rect#cell-id-{cell_id}'
            cell_g = soup.select_one(f'g.xwd__cell:has({cell_selector})')

            letter_found = None
            if cell_g:
                text_tags = cell_g.find_all('text')
                for text_tag in text_tags:
                    letter = text_tag.get_text(strip=True)
                    if len(letter) == 1 and letter.isalpha():
                        letter_found = letter.upper()
                        break

            if letter_found:
                answer_letters.append(letter_found)
            else:
                answer_letters = []  # Partially filled — invalidate
                break

            if direction == 'across':
                c += 1
            else:
                r += 1

        end_r, end_c = cells_in_clue[-1]
        length = len(cells_in_clue)

        row = {
            "number": number,
            "start_col": start_c,
            "start_row": start_r,
            "end_col": end_c,
            "end_row": end_r,
            "clue": clue_text,
            "length (optional column, for checking only)": length
        }

        if len(answer_letters) == length:
            row["answer (optional column, for checking only)"] = ''.join(answer_letters)
            answers_available = True

        full_output.append(row)

    # Step 5: Build DataFrame
    required_columns = ["number", "start_col", "start_row", "end_col", "end_row", "clue"]
    optional_columns = ["length (optional column, for checking only)"]
    if answers_available:
        optional_columns.append("answer (optional column, for checking only)")

    clue_df = pd.DataFrame(full_output)
    clue_df = clue_df[required_columns + optional_columns]

    return clue_df


def get_random_clue_df(folder=r"data/puzzle_samples/raw_html/", return_type="All"):
    """
    Get a parsed clue_df from a puzzle HTML file in the given folder.

    genai.

    :param folder: Directory where HTML files are stored
    :param return_type: "All", "All regular", "All minis", "Random regular", or "Random mini"
    :return: clue_df parsed from a puzzle or list of clue_dfs
    """

    # If folder is the default, then create full path
    if folder == r"data/puzzle_samples/raw_html/":
        folder = os.path.join(get_project_root(), folder)

    # Normalize return type
    return_type = return_type.lower()

    # Get list of all html files in given folder
    all_files = [f for f in os.listdir(folder) if f.endswith('.html')]
    if not all_files:
        raise FileNotFoundError(f"No HTML files found in {folder}")

    # Apply filtering if needed
    filtered = []
    if "mini" in return_type:
        filtered = [f for f in all_files if "mini" in f.lower()]

    elif "regular" in return_type:
        filtered = [f for f in all_files if "mini" not in f.lower()]

    elif return_type == "all":
        filtered = all_files

    else:
        raise ValueError(f"Invalid return_type: '{return_type}'.")

    result_dict = {}

    # Random single puzzle as df
    if "random" in return_type:
        selected = random.choice(filtered)
        full_path = os.path.join(get_project_root(), folder, selected)
        result_dict[selected[:-5]] = puzzle_html_to_df(full_path)

    # Return all as list of DataFrames
    if "all" in return_type:
        result_dict = {}
        for file in filtered:
            full_path = os.path.join(get_project_root(), folder, file)
            print(full_path)
            df = puzzle_html_to_df(full_path)
            result_dict[file[:-5]] = df

    return result_dict


def process_all_raw_html_to_csv(overwrite=False):
    """
    Calls get_random_clue_df (which converts all html files from raw_html into clue_df
    format). Then saves each file as a csv.
    :return: True if no errors arise
    """
    print("Getting all dataframe from raw html files")
    all_clue_dfs = get_random_clue_df(return_type="all")

    save_folder = fr"{get_project_root()}/data/puzzle_samples/processed_puzzle_samples"

    for puzzle_name in all_clue_dfs.keys():
        save_path = fr"{save_folder}/{puzzle_name}.csv"
        if os.path.exists(save_path) and overwrite is False:
            print(f"Not overwriting csv because file already exists and overwright=False: {save_path}")
            continue
        else:
            print(save_path)
            clue_df = all_clue_dfs[puzzle_name]
            print(clue_df)
            clue_df.to_csv(save_path, index=False)

    return True

def rename_puzzles():
    """
    Chatgpt function. This just renames from the standard format that puzzles download as into a more easily
    readable / parsable format.

    :return: nothing
    """
    raw_puzzle_loc = f"{get_project_root()}/data/puzzle_samples/raw_html"
    pattern = re.compile(r"^(?P<day>\w+), (?P<month>\w+) (?P<day_num>\d{1,2}),"
                         r" (?P<year>\d{4}) The (?P<type>Mini|Crossword) puzzle — The New York Times\.html$")

    for filename in os.listdir(raw_puzzle_loc):
        match = pattern.match(filename)
        if match:
            month_str = match.group("month")
            try:
                month_num = datetime.strptime(month_str, "%B").month
            except ValueError:
                print(f"Skipping: Invalid month in filename: {filename}")
                continue

            year = match.group("year")
            day = match.group("day_num").zfill(2)
            month = str(month_num).zfill(2)
            puzzle_type = match.group("type").lower()  # "crossword" or "mini"
            new_name = f"{puzzle_type}_{year}_{month}_{day}.html"

            old_path = os.path.join(raw_puzzle_loc, filename)
            new_path = os.path.join(raw_puzzle_loc, new_name)

            if not os.path.exists(new_path):  # avoid accidental overwrite
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} → {new_name}")
            else:
                print(f"Skipped (already exists): {new_name}")


#mini_loc = f"{get_project_root()}/data/puzzle_samples/raw_html/mini.html"
#clue_df = puzzle_html_to_df(mini_loc)
#my_crossword = Crossword(clue_df=clue_df)


# proj_root = get_project_root()
# print(f"proj_root:{proj_root}")
# html_path = fr"{proj_root}/data/puzzle_samples/raw_html/mini_03262025.html"
# print(f"html_path:{html_path}")
# clue_df = puzzle_html_to_df(html_path)
# clue_df.to_csv(r"C:\Users\witzi\OneDrive\Documents\neu_part_2\CS5100_FAI\code\ai_crossword_solver\data\puzzle_samples\sunday_03092025.csv")
