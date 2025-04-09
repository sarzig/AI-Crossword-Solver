import csv
import re
import os.path
from difflib import get_close_matches

class CrosswordSolver:
    def __init__(self, csv_file):
        """
        Initialize the CrosswordSolver with a CSV file containing crossword clues and answers.
        
        Args:
            csv_file (str): Path to the CSV file with columns: Date, Word, Clue, Detected_Language
        """
        self.clue_to_word = {}
        self.word_to_clue = {}
        self.all_clues = []  # Store all clues for better searching
        self.languages = set()
        
        if not os.path.exists(csv_file):
            print(f"Error: File '{csv_file}' not found.")
            return
            
        try:
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    clue = row['Clue'].strip()
                    word = row['Word'].strip().upper()
                    language = row['Detected_Language'].strip()
                    
                    # Store in our dictionaries - no normalization to preserve exact format
                    self.clue_to_word[clue] = word
                    self.word_to_clue[word] = clue
                    self.all_clues.append((clue, word))
                    self.languages.add(language)
            
            # Print first 5 clues for debugging
            print(f"Loaded {len(self.clue_to_word)} clues from {csv_file}")
            print(f"Languages detected: {', '.join(sorted(self.languages))}")
            print("\nSample clues (first 5):")
            for i, (clue, word) in enumerate(list(self.clue_to_word.items())[:5], 1):
                print(f"{i}. '{clue}' -> {word}")
            
        except Exception as e:
            print(f"Error loading CSV file: {e}")

    def find_exact_match(self, clue):
        """
        Find an exact match for the given clue.
        
        Args:
            clue (str): The clue to search for
            
        Returns:
            str or None: The matching word, or None if not found
        """
        # Show what we're searching for
        print(f"\nSearching for exact match: '{clue}'")
        
        # Try exact match first
        if clue in self.clue_to_word:
            return self.clue_to_word[clue]
        
        # Debug: print some clues from the dictionary to see why it's not matching
        print("No exact match found. Here are some clues from the database:")
        clue_list = list(self.clue_to_word.keys())
        for i, db_clue in enumerate(clue_list[:5], 1):
            print(f"{i}. '{db_clue}'")
            
        # Check if there's a close match (might be whitespace or quotes issue)
        for db_clue in self.clue_to_word:
            # Compare with special characters and whitespace removed
            db_norm = re.sub(r'[^\w\s]', '', db_clue).lower().strip()
            input_norm = re.sub(r'[^\w\s]', '', clue).lower().strip()
            
            if db_norm == input_norm:
                print(f"Found close match: '{db_clue}'")
                return self.clue_to_word[db_clue]
                
        return None
    
    def find_partial_matches(self, clue, max_results=5):
        """
        Find clues that partially match the given clue.
        
        Args:
            clue (str): The clue to search for
            max_results (int): Maximum number of results to return
            
        Returns:
            list: List of (word, clue) tuples for partial matches
        """
        print(f"\nSearching for partial matches for: '{clue}'")
        matches = []
        
        # Extract important terms from the clue
        search_terms = []
        
        # Handle quotes in clues - they're often important parts
        quoted_phrases = re.findall(r'"([^"]*)"', clue)
        search_terms.extend(quoted_phrases)
        
        # Look for language indicators (e.g., "in Spanish", "in French")
        language_match = re.search(r'in (\w+)', clue)
        if language_match:
            search_terms.append(language_match.group(0))  # Add the "in X" phrase
        
        # Extract keywords (words longer than 3 letters)
        keywords = [word for word in re.findall(r'\b\w+\b', clue) if len(word) > 3]
        search_terms.extend(keywords)
        
        # Remove duplicates
        search_terms = list(set(search_terms))
        
        print(f"Search terms extracted: {search_terms}")
        
        # Search for each term in the clues
        for db_clue, word in self.all_clues:
            score = 0
            matches_found = []
            
            # Check if search terms appear in the database clue
            for term in search_terms:
                if term.lower() in db_clue.lower():
                    score += 1
                    matches_found.append(term)
            
            # If we have a good match (at least one term appears)
            if score > 0:
                matches.append((word, db_clue, score, matches_found))
        
        # Sort by score (descending) and return top matches
        matches.sort(key=lambda x: x[2], reverse=True)
        
        # Debug: print what we found
        if matches:
            print(f"Found {len(matches)} partial matches. Top matches:")
            for i, (word, db_clue, score, terms) in enumerate(matches[:3], 1):
                print(f"{i}. '{db_clue}' (Score: {score}, Matched terms: {terms})")
        else:
            print("No partial matches found.")
            
            # Let's do a broader search to help debug
            print("\nBroadening search... Looking for any clues containing key words:")
            for term in search_terms:
                if len(term) > 3:  # Only check substantial terms
                    found = False
                    for db_clue, word in self.all_clues:
                        if term.lower() in db_clue.lower():
                            print(f"Found '{term}' in: '{db_clue}' -> {word}")
                            found = True
                            break
                    if not found:
                        print(f"No clues contain the term '{term}'")
        
        return [(word, clue) for word, clue, _, _ in matches[:max_results]]
    
    def find_by_pattern(self, pattern_input):
        """
        Find words or clues matching a pattern.
        
        This can handle:
        1. Word patterns like "A__E" or "A**E" where _ or * is any letter
        2. Full clue text (will search for the clue)
        
        Args:
            pattern_input (str): Pattern to search for
            
        Returns:
            list: List of (word, clue) tuples matching the pattern
        """
        print(f"\nPattern search input: '{pattern_input}'")
        matches = []
        
        # If input contains quotes or comma or parentheses, it's probably a clue not a pattern
        if '"' in pattern_input or ',' in pattern_input or '(' in pattern_input:
            print("Input appears to be a clue rather than a word pattern.")
            # Search as a clue instead
            exact_match = self.find_exact_match(pattern_input)
            if exact_match:
                matches.append((exact_match, pattern_input))
            else:
                matches = self.find_partial_matches(pattern_input)
            return matches
            
        # Check if input looks like a word pattern (contains * or _)
        if '_' in pattern_input or '*' in pattern_input:
            # Convert * to _ for internal consistency
            pattern = pattern_input.replace('*', '_').upper()
            
            # Create regex pattern for matching
            regex_pattern = '^' + pattern.replace('_', '.') + '$'
            print(f"Using regex pattern: {regex_pattern}")
            
            # Search for matching words
            for word, clue in self.word_to_clue.items():
                if re.match(regex_pattern, word):
                    matches.append((word, clue))
                    
            print(f"Found {len(matches)} matches for pattern {pattern}")
        else:
            # Treat as a clue search
            exact_match = self.find_exact_match(pattern_input)
            if exact_match:
                matches.append((exact_match, pattern_input))
            else:
                # Try partial match
                matches = self.find_partial_matches(pattern_input)
                
        return matches

    def search_by_answer(self, answer):
        """
        Find clues for a specific answer word.
        
        Args:
            answer (str): The answer word to look up
            
        Returns:
            list: List of clues for this answer
        """
        answer = answer.upper().strip()
        matches = []
        
        for word, clue in self.word_to_clue.items():
            if word == answer:
                matches.append(clue)
                
        return matches

def main():
    # Use the exact file path provided by the user
    import os
    # Direct path to the CSV file
    csv_file ='foreign_language_clues_with_language.csv'
    
    print(f"Looking for CSV file at: {csv_file}")
    
    # Check if file exists before initializing
    #if not os.path.exists(csv_file):
       ## print(f"ERROR: CSV file not found at {csv_file}")
        #print("Please make sure the file is in the correct location and named correctly.")
        #print("Current working directory is:", current_dir)
        #print("\nAvailable files in current directory:")
        #for file in os.listdir(current_dir):
            #print(f"- {file}")
       # return
        
    # Initialize the solver with the CSV file
    solver = CrosswordSolver(csv_file)
    
    while True:
        print("\nCrossword Puzzle Solver")
        print("=" * 25)
        print("1. Search by clue")
        print("2. Search by word pattern (use _ or * for unknown letters)")
        print("3. Search by answer word")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            clue = input("Enter the clue: ").strip()
            
            # Try exact match first
            exact_match = solver.find_exact_match(clue)
            if exact_match:
                print(f"\nExact match found: {exact_match}")
            else:
                print("\nNo exact match found. Searching for partial matches...")
                partial_matches = solver.find_partial_matches(clue)
                
                if partial_matches:
                    print("\nPossible matches:")
                    for i, (word, matched_clue) in enumerate(partial_matches, 1):
                        print(f"{i}. {word} - {matched_clue}")
                else:
                    print("No matches found.")
                    
        elif choice == '2':
            pattern = input("Enter word pattern (use _ or * for unknown letters) or a clue: ").strip()
            matches = solver.find_by_pattern(pattern)
            
            if matches:
                print(f"\nFound {len(matches)} matches:")
                for i, (word, clue) in enumerate(matches, 1):
                    print(f"{i}. {word} - {clue}")
            else:
                print("\nNo matches found for that pattern or clue.")
        
        elif choice == '3':
            answer = input("Enter the answer word: ").strip()
            clues = solver.search_by_answer(answer)
            
            if clues:
                print(f"\nFound {len(clues)} clues for '{answer}':")
                for i, clue in enumerate(clues, 1):
                    print(f"{i}. {clue}")
            else:
                print(f"\nNo clues found for '{answer}'.")
                
        elif choice == '4':
            print("Thank you for using the Crossword Puzzle Solver!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()