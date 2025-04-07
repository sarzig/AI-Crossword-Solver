def auto_place_clues(cw, clue_answer_map, verbose=True):
    """
    Places words in the crossword for a provided mapping of clue variable → answer.

    Parameters:
    - cw: Crossword object
    - clue_answer_map: dict of format { "1-Across": "apple", "3-Down": "zebra", ... }
    - verbose: if True, prints placement results
    """
    placed = []
    failed = []

    for var, answer in clue_answer_map.items():
        try:
            answer = answer.strip().lower()
            cw.place_word(answer, var)
            placed.append((var, answer))
            if verbose:
                print(f"✅ Placed: {var} → '{answer}'")
        except Exception as e:
            failed.append((var, answer, str(e)))
            if verbose:
                print(f"❌ Failed to place: {var} → '{answer}' | Error: {e}")

    print(f"\n📌 Summary:")
    print(f"  ✅ Placed {len(placed)} clues successfully.")
    print(f"  ❌ Failed to place {len(failed)} clues.")

    return placed, failed