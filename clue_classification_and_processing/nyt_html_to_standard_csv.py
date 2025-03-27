import re
import pandas as pd
from bs4 import BeautifulSoup

"""
This entire parsing was created with ChatGPT.

Using saved html, this converts a crossword into a meaningful dataframe / saves
to csv.

xxx tbd - needs to account for answers optionally! Needs correct length column name
needs to output all info to df (currently does subset)
"""


def get_coords(x, y, cell_size, cell_offset):
    col = round((float(x) - cell_offset) / cell_size)
    row = round((float(y) - cell_offset) / cell_size)
    return row, col


def puzzle_html_to_df(filename):

    # Load HTML
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
    cells = []
    grid_map = {}
    max_row, max_col = 0, 0

    # Find the cell size and cell offset programatically (Diff for different screen size and mini vs. regular)
    # Step 0: Extract all cell rects and get their (x, y)
    cell_rects = soup.select('g.xwd__cell rect')
    positions = []

    for rect in cell_rects:
        if 'x' in rect.attrs and 'y' in rect.attrs:
            x = float(rect['x'])
            y = float(rect['y'])
            positions.append((x, y))

    # Step 0.5: Sort and deduplicate
    positions = sorted(set(positions))

    # Step 0.6: Infer cell_offset (min x or y)
    min_x = min(pos[0] for pos in positions)
    min_y = min(pos[1] for pos in positions)
    cell_offset = min(min_x, min_y)

    # Step 0.7: Infer cell_size from first row (same y)
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
        row, col = get_coords(x, y, cell_size=cell_size, cell_offset=cell_offset)

        max_row = max(max_row, row)
        max_col = max(max_col, col)

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

    # Step 4: For each clue, trace it across/down
    output = []
    for clue in clue_list:
        number = clue['number']
        direction = clue['direction']
        clue_text = clue['clue']

        if number not in number_to_position:
            output.append(f"{number}\t-1\t-1\t-1\t-1\t{clue_text}\t0\t{direction}")
            continue

        start_r, start_c = number_to_position[number]
        r, c = start_r, start_c
        cells_in_clue = []

        while (r, c) in grid_map and not grid_map[(r, c)]['is_black']:
            cells_in_clue.append((r, c))
            if direction == 'across':
                c += 1
            else:
                r += 1

        end_r, end_c = cells_in_clue[-1]
        length = len(cells_in_clue)

        output.append(f"{number}\t{start_c}\t{start_r}\t{end_c}\t{end_r}\t{clue_text}\t{length}\t{direction}")

    # Final output
    print("number\tstart_col\tstart_row\tend_col\tend_row\tclue\tlength\tdirection")
    for line in sorted(output, key=lambda l: int(l.split('\t')[0])):
        print(line)

    clue_df = pd.DataFrame(clue_list)

    return clue_df


file = r"C:\Users\witzi\OneDrive\Documents\neu_part_2\CS5100_FAI\code\ai_crossword_solver\data\puzzle_samples\raw_html\mini_03262025.html"
clue_df = puzzle_html_to_df(file)
#clue_df.to_csv(r"C:\Users\witzi\OneDrive\Documents\neu_part_2\CS5100_FAI\code\ai_crossword_solver\data\puzzle_samples\sunday_03092025.csv")