import re


def extract_clues(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Regular expressions to extract across and down clues
    across_clue_pattern = re.compile(
        r'<li class="xwd__clue--li"><span class="xwd__clue--label">(\d+)</span><span class="xwd__clue--text xwd__clue-format">(.*?)</span>')
    down_clue_pattern = re.compile(
        r'<li class="xwd__clue--li xwd__clue--highlighted"><span class="xwd__clue--label">(\d+)</span><span class="xwd__clue--text xwd__clue-format">(.*?)</span>')

    across_clues = across_clue_pattern.findall(html_content)
    down_clues = down_clue_pattern.findall(html_content)

    return across_clues, down_clues


# File path to the crossword HTML
file_path = "crossword_html.txt"
across_clues, down_clues = extract_clues(file_path)

# Display the extracted clues
print("Across Clues:")
for num, clue in across_clues:
    print(f"{num}. {clue}")

print("\nDown Clues:")
for num, clue in down_clues:
    print(f"{num}. {clue}")
