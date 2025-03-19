import requests
import pandas as pd

######################################################################################################
# Get one layer of synonyms
######################################################################################################

def get_synonyms(word):
    """Fetch synonyms for a given word using the Datamuse API."""
    url = f"https://api.datamuse.com/words?rel_syn={word}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return {item['word'] for item in data}  # Return a set to avoid duplicates
    else:
        return set()

######################################################################################################
# Get recursive/multi-layer synonyms
######################################################################################################

def get_recursive_synonyms(word, depth):
    """
    Recursively fetch synonyms up to a given depth, including original synonyms.
    
    depth=1 → Only direct synonyms
    depth=2 → Synonyms + synonyms of synonyms (default)
    depth=3 → Goes one level deeper, and so on
    """
    all_synonyms = set()
    queue = {word}  # Start with the input word

    for _ in range(depth):
        new_synonyms = set()
        for w in queue:
            syns = get_synonyms(w)
            new_synonyms.update(syns)
        
        # Keep original synonyms while avoiding duplicate API calls
        all_synonyms.update(queue)  
        queue = new_synonyms - all_synonyms  # Only new words for next iteration

    all_synonyms.update(queue)  # Add final level words
    return sorted(all_synonyms)  # Return sorted list

######################################################################################################
# Example Usage
######################################################################################################

word = input("Enter a word: ")
synonyms = get_recursive_synonyms(word, depth=3)
print(f"Expanded synonyms for '{word}': {synonyms}")


######################################################################################################
# Testing on the list of synonym clues
######################################################################################################

# Load the data set
csv_synonyms_path = 'single_word_clues(Sheet1).csv'
df_test = pd.read_csv(csv_synonyms_path, encoding="ISO-8859-1")

df_test_sample = df_test.sample(n = 100, random_state=42)

# Ensure correct column names (adjust if needed)
df_test.columns = ["Date", "Word", "Clue"]  # Rename columns for clarity


def check_synonym_accuracy(df):
    """ Check if crossword answers appear in the synonyms retrieved from Datamuse. """
    results = []

    for index, row in df.iterrows():
        clue = row["Clue"].strip().lower()  # Normalize clue
        answer = row["Word"].strip().lower()  # Normalize answer

        synonyms = get_recursive_synonyms(clue, 4)  # Fetch synonyms
        answer_found = answer in synonyms  # Check if answer is in the synonyms list

        results.append({
            "Clue": clue,
            "Expected Answer": answer,
            "Synonyms": ", ".join(sorted(synonyms)) if synonyms else "No synonyms found",
            "Answer Found in Synonyms": answer_found
        })

        print(f"{index+1}/{len(df)} | Clue: {clue} | Expected Answer: {answer} | Found: {answer_found}")

        # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate accuracy
    accuracy = results_df["Answer Found in Synonyms"].mean() * 100
    print(f"\n Synonym Match Accuracy: {accuracy:.2f}% ({results_df['Answer Found in Synonyms'].sum()} out of {len(df)})")

    # Save results to CSV
    output_file = "synonym_search_results.csv"
    results_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\n Results saved to: {output_file}")

    # Display results
    return results_df

######################################################################################################
# Usage of synonym accuracy check
######################################################################################################

# Run the synonym accuracy check
# results_df = check_synonym_accuracy(df_test_sample)

# print(f"Results: {results_df}")
