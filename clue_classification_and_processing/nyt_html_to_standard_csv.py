import pandas as pd

# Step 5: Convert final clue data to DataFrame and save to CSV
output_rows = []

for clue in clue_list:
    number = clue['number']
    direction = clue['direction']
    clue_text = clue['clue']

    if number not in number_to_position:
        output_rows.append({
            'number': number,
            'start_col': -1,
            'start_row': -1,
            'end_col': -1,
            'end_row': -1,
            'clue': clue_text,
            'length (optional column, for checking only)': 0
        })
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

    output_rows.append({
        'number': number,
        'start_col': start_c,
        'start_row': start_r,
        'end_col': end_c,
        'end_row': end_r,
        'clue': clue_text,
        'length (optional column, for checking only)': length
    })

# Save to CSV
df = pd.DataFrame(output_rows)
df.to_csv(
    r"C:\Users\witzi\OneDrive\Documents\neu_part_2\CS5100_FAI\code\ai_crossword_solver\data\puzzle_samples\sunday_03092025.csv",
    index=False
)
