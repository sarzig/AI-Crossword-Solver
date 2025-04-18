def auto_place_clues(cw, clue_answer_map, verbose=False):
    placed = []
    failed = []

    # Create a set of valid clue labels from the crossword
    valid_clues = set(cw.clue_df["number_direction"].dropna().astype(str))

    for var, answer in clue_answer_map.items():
        if var not in valid_clues:
            failed.append((var, answer, "Clue not found in crossword"))
            continue

        try:
            answer = answer.strip().lower()
            cw.place_word(answer, var)
            placed.append((var, answer))
        except Exception as e:
            failed.append((var, answer, str(e)))

    if verbose:
        if placed:
            print("\n✅ Successfully Placed:")
            print("  " + " | ".join([f"{var}: '{ans}'" for var, ans in placed]))

        if failed:
            print("\n❌ Failed to Place:")
            for var, ans, err in failed:
                print(f"  {var}: '{ans}' | Error: {err}")

    print(f"\n📌 Summary:")
    print(f"  ✅ Placed {len(placed)} clues successfully.")
    print(f"  ❌ Failed to place {len(failed)} clues.")

    return placed, failed
