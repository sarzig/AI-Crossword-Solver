import pandas as pd
import re
import sys
import requests
from bs4 import BeautifulSoup
import time
import random

class AcronymLookup:
    def __init__(self, csv_file=None):
        """
        Initialize the AcronymLookup with a CSV file containing acronym data.
        
        Args:
            csv_file (str, optional): Path to the CSV file with columns: Date, Word, Clue.
                                     If None, uses default path.
        """
        # Default path for macOS
        if csv_file is None:
            csv_file = "/Users/swathibabaeswarappa/Downloads/FAI /nyt_acronym_clues.csv"
        
        self.df = pd.read_csv(csv_file)
        
        # Create indices for faster lookup
        self.word_to_clue_dict = self._create_word_to_clue_dict()
        self.clue_to_word_dict = self._create_clue_to_word_dict()
        
        # Create pattern-based indices for fuzzy matching
        self._create_pattern_indices()
        
        # Cache for web requests to avoid repeated scraping
        self.web_cache = {}
        
        # Common acronyms dictionary for offline mode or backup
        self._create_common_acronyms_dict()
    
    def _create_word_to_clue_dict(self):
        """Create a dictionary mapping words to their clues."""
        word_to_clue = {}
        for _, row in self.df.iterrows():
            word = row['Word'].strip().upper()
            clue = row['Clue'].strip()
            
            if word not in word_to_clue:
                word_to_clue[word] = []
            
            if clue not in word_to_clue[word]:
                word_to_clue[word].append(clue)
        
        return word_to_clue
    
    def _create_clue_to_word_dict(self):
        """Create a dictionary mapping clues to their words."""
        clue_to_word = {}
        for _, row in self.df.iterrows():
            word = row['Word'].strip().upper()
            clue = row['Clue'].strip()
            
            if clue not in clue_to_word:
                clue_to_word[clue] = []
            
            if word not in clue_to_word[clue]:
                clue_to_word[clue].append(word)
        
        return clue_to_word
    
    def _create_pattern_indices(self):
        """Create pattern-based indices for fuzzy matching."""
        # Create keyword indices for clues
        self.keyword_to_clue = {}
        for clue in self.clue_to_word_dict.keys():
            # Extract keywords from clue
            keywords = re.findall(r'\b\w+\b', clue.lower())
            for keyword in keywords:
                if len(keyword) > 2:  # Ignore very short words
                    if keyword not in self.keyword_to_clue:
                        self.keyword_to_clue[keyword] = []
                    self.keyword_to_clue[keyword].append(clue)
    
    def _create_common_acronyms_dict(self):
        """Create a dictionary of common acronyms as a backup."""
        self.common_acronyms = {
            "NASA": ["National Aeronautics and Space Administration", "US space agency"],
            "HTML": ["Hypertext Markup Language", "Web page coding language"],
            "CSS": ["Cascading Style Sheets", "Web styling language"],
            "JSON": ["JavaScript Object Notation", "Data interchange format"],
            "API": ["Application Programming Interface", "Software connection interface"],
            "SQL": ["Structured Query Language", "Database query language"],
            "XML": ["Extensible Markup Language", "Markup language"],
            "GPS": ["Global Positioning System", "Navigation satellite system"],
            "FBI": ["Federal Bureau of Investigation", "US federal law enforcement agency"],
            "CIA": ["Central Intelligence Agency", "US intelligence agency"],
            "UN": ["United Nations", "International organization"],
            "USB": ["Universal Serial Bus", "Computer connection standard"],
            "URL": ["Uniform Resource Locator", "Web address"],
            "ASAP": ["As Soon As Possible", "Without delay"],
            "DIY": ["Do It Yourself", "Self-made projects"],
            "ATM": ["Automated Teller Machine", "Cash dispenser"],
            "DVD": ["Digital Versatile Disc", "Optical storage medium"],
            "TV": ["Television", "Broadcast video device"],
            "PC": ["Personal Computer", "Individual computing device"],
            "CEO": ["Chief Executive Officer", "Top company executive"],
            "RSVP": ["Répondez S'il Vous Plaît", "Please respond to invitation"],
            "WIFI": ["Wireless Fidelity", "Wireless internet connection"],
            "RADAR": ["Radio Detection And Ranging", "Object detection system"],
            "LASER": ["Light Amplification by Stimulated Emission of Radiation", "Focused light beam"],
            "SCUBA": ["Self-Contained Underwater Breathing Apparatus", "Diving equipment"],
            "SONAR": ["Sound Navigation And Ranging", "Underwater object detection"],
            "CAPTCHA": ["Completely Automated Public Turing test to tell Computers and Humans Apart", "Bot detection system"]
        }
    
    def _scrape_acronym_finder(self, acronym):
        """
        Scrape Acronym Finder website for the given acronym.
        
        Args:
            acronym (str): The acronym to search for
            
        Returns:
            list: List of definitions found for the acronym
        """
        # Check cache first
        if acronym in self.web_cache:
            return self.web_cache[acronym]
        
        try:
            # Construct the URL for the acronym
            url = f"https://www.acronymfinder.com/{acronym}.html"
            print(f"Searching online at: {url}")
            
            # Add a user agent to mimic a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Make the request
            response = requests.get(url, headers=headers, timeout=10)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Parse the HTML content
                soup = BeautifulSoup(response.text, 'html.parser')
                print(f"Successfully retrieved data for {acronym}")
                
                # Find the definition table
                results = []
                
                # Search for various table formats that might contain acronym definitions
                tables = soup.find_all('table')
                definition_table = None
                
                for table in tables:
                    if 'table-striped' in table.get('class', []) or 'table' in table.get('class', []):
                        definition_table = table
                        break
                
                if definition_table:
                    print("Found definition table")
                    rows = definition_table.find_all('tr')
                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) >= 2:
                            # Extract the meaning
                            meaning = cols[1].get_text(strip=True)
                            if meaning:
                                results.append(meaning)
                else:
                    # Also try looking for definitions in a different format
                    definition_divs = soup.select('.block-acronym')
                    if definition_divs:
                        print("Found definition blocks")
                        for div in definition_divs:
                            meaning_element = div.select_one('.meaning')
                            if meaning_element:
                                meaning = meaning_element.get_text(strip=True)
                                if meaning:
                                    results.append(meaning)
                
                if not results:
                    # Try a more general approach to find definitions
                    print("Trying general approach to find definitions")
                    # Look for any elements that might contain definitions
                    possible_definitions = soup.find_all(['div', 'p', 'li'], class_=['definition', 'meaning', 'result'])
                    for element in possible_definitions:
                        text = element.get_text(strip=True)
                        if text and len(text) > 5:  # Avoid very short texts
                            results.append(text)
                
                # Print debug info
                print(f"Found {len(results)} definitions for {acronym}")
                
                # Add to cache to avoid repeated requests
                self.web_cache[acronym] = results
                
                # Add a small delay to be respectful to the server
                time.sleep(random.uniform(0.5, 1.5))
                
                return results
            else:
                print(f"Failed to retrieve data for {acronym}. Status code: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error scraping data for {acronym}: {str(e)}")
            return []
    
    def _search_acronym_expanded(self, acronym):
        """
        Get the expanded form of an acronym from multiple sources.
        
        Args:
            acronym (str): The acronym to expand
            
        Returns:
            list: List of possible expansions
        """
        # First try web scraping
        web_results = self._scrape_acronym_finder(acronym)
        
        # If we got results from the web, return them
        if web_results:
            return [(result, "web") for result in web_results]
        
        # Otherwise check our common acronyms dictionary
        if acronym in self.common_acronyms:
            return [(result, "common") for result in self.common_acronyms[acronym]]
        
        # If nothing found, try alternate sources or return empty list
        return []
    
    def _search_reverse_acronym_finder(self, query):
        """
        Search for acronyms that match a given phrase.
        
        Args:
            query (str): The phrase to search for
            
        Returns:
            list: List of acronyms found for the query
        """
        # Check cache first
        cache_key = f"reverse_{query}"
        if cache_key in self.web_cache:
            return self.web_cache[cache_key]
        
        try:
            # Format the query for URL
            formatted_query = query.replace(' ', '+')
            url = f"https://www.acronymfinder.com/~/search/aj/?a=search&st={formatted_query}"
            print(f"Searching online for reverse lookup: {url}")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'application/json, text/javascript, */*; q=0.01',
                'X-Requested-With': 'XMLHttpRequest'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                try:
                    # The response might be JSON or HTML, try JSON first
                    results = []
                    data = response.json()
                    
                    if 'results' in data and data['results']:
                        for item in data['results']:
                            if 'a' in item and 't' in item:  # Acronym and term
                                results.append(f"{item['a']} - {item['t']}")
                                
                        # Add to cache
                        self.web_cache[cache_key] = results
                        
                        # Add a small delay
                        time.sleep(random.uniform(0.5, 1.5))
                        
                        return results
                except:
                    # If JSON fails, try parsing HTML
                    soup = BeautifulSoup(response.text, 'html.parser')
                    results = []
                    
                    # Find result items - this depends on the actual structure of the page
                    items = soup.select('.acronym-item')  # Adjust selector based on actual page
                    
                    for item in items:
                        acronym = item.select_one('.acronym')
                        meaning = item.select_one('.meaning')
                        if acronym and meaning:
                            results.append(f"{acronym.get_text(strip=True)} - {meaning.get_text(strip=True)}")
                    
                    # Add to cache
                    self.web_cache[cache_key] = results
                    
                    # Add a small delay
                    time.sleep(random.uniform(0.5, 1.5))
                    
                    return results
            
            return []
        except Exception as e:
            print(f"Error in reverse search for {query}: {str(e)}")
            return []
    
    def get_clues_for_acronym(self, acronym):
        """
        Get all clues and definitions for a given acronym.
        
        Args:
            acronym (str): The acronym to look up
            
        Returns:
            list: List of clues and definitions associated with the acronym
        """
        acronym = acronym.strip().upper()
        
        # First check our local dataset
        dataset_clues = []
        if acronym in self.word_to_clue_dict:
            dataset_clues = self.word_to_clue_dict[acronym]
        
        # Get expanded forms of the acronym
        expanded_forms = self._search_acronym_expanded(acronym)
        
        # If no expanded forms found but we have dataset clues, just return those
        if not expanded_forms and dataset_clues:
            return dataset_clues
        
        # If we have expanded forms, add them to the results
        result_clues = []
        
        # Add expanded forms first
        for expansion, source in expanded_forms:
            if source == "web":
                result_clues.append(f"{expansion} (from web)")
            else:
                result_clues.append(f"{expansion} (common definition)")
        
        # Then add dataset clues
        for clue in dataset_clues:
            result_clues.append(clue)
        
        # If we still have no results, try to find similar acronyms
        if not result_clues:
            similar_acronyms = []
            for word in self.word_to_clue_dict.keys():
                if len(word) == len(acronym):
                    # Calculate similarity
                    matches = sum(1 for a, b in zip(word, acronym) if a == b)
                    if matches >= len(acronym) - 1:  # Allow 1 character difference
                        similar_acronyms.append(word)
            
            if similar_acronyms:
                suggestions = []
                for similar in similar_acronyms[:3]:  # Limit to top 3
                    clues = self.word_to_clue_dict[similar]
                    first_clue = clues[0] if clues else "Unknown"
                    suggestions.append(f"{similar} ({first_clue})")
                
                return [f"Did you mean: {', '.join(suggestions)}?"]
            
            return ["No definitions found for this acronym."]
        
        return result_clues
    
    def get_acronyms_for_clue(self, clue):
        """
        Get all acronyms for a given clue.
        
        Args:
            clue (str): The clue to look up
            
        Returns:
            list: List of acronyms associated with the clue
        """
        clue = clue.strip()

        # NEW CODE: Check if this clue matches any definitions in web results/common acronyms
        clue_lower = clue.lower()
        
        # Check the web cache for matching definitions
        reverse_matches = []
        for acronym, definitions in self.web_cache.items():
            if isinstance(definitions, list):
                for definition in definitions:
                    # Clean the definition text
                    clean_def = definition.replace(" (from web)", "").strip().lower()
                    # Check for exact or close matches
                    if clue_lower == clean_def or clue_lower in clean_def or clean_def in clue_lower:
                        reverse_matches.append(f"{acronym} (matches web definition: '{definition}')")
        
        # Also check common acronyms dictionary
        for acronym, definitions in self.common_acronyms.items():
            for definition in definitions:
                def_lower = definition.lower()
                if clue_lower == def_lower or clue_lower in def_lower or def_lower in clue_lower:
                    reverse_matches.append(f"{acronym} (matches definition: '{definition}')")
        
        # If we found matches from definitions, return them
        if reverse_matches:
            return reverse_matches
        
        # EXISTING CODE CONTINUES FROM HERE:
        # Check for exact match in dataset
        if clue in self.clue_to_word_dict:
            return self.clue_to_word_dict[clue]
        
        # Try fuzzy matching in dataset
        potential_matches = []
        keywords = re.findall(r'\b\w+\b', clue.lower())
        matched_clues = set()
        


        # Check for exact match in dataset
        if clue in self.clue_to_word_dict:
            return self.clue_to_word_dict[clue]
        
        # Try fuzzy matching in dataset
        potential_matches = []
        keywords = re.findall(r'\b\w+\b', clue.lower())
        matched_clues = set()
        
        for keyword in keywords:
            if len(keyword) > 2 and keyword in self.keyword_to_clue:
                for matched_clue in self.keyword_to_clue[keyword]:
                    if matched_clue not in matched_clues:
                        matched_clues.add(matched_clue)
                        # Calculate similarity score (simple overlap for now)
                        matched_keywords = set(re.findall(r'\b\w+\b', matched_clue.lower()))
                        query_keywords = set(keywords)
                        overlap = len(matched_keywords.intersection(query_keywords))
                        
                        if overlap > 0:
                            # Get the acronyms for this clue
                            acronyms = self.clue_to_word_dict[matched_clue]
                            
                            for acronym in acronyms:
                                potential_matches.append({
                                    'acronym': acronym,
                                    'clue': matched_clue,
                                    'score': overlap
                                })
        
        # Sort by score
        potential_matches.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top unique acronyms from fuzzy matching
        unique_acronyms = []
        seen = set()
        for match in potential_matches:
            if match['acronym'] not in seen and len(unique_acronyms) < 5:
                unique_acronyms.append(match['acronym'])
                seen.add(match['acronym'])
        
        # If we found matches in our dataset, return them
        if unique_acronyms:
            return unique_acronyms
        
        # If no matches in dataset, try web scraping
        print(f"Clue not found in dataset. Searching online...")
        web_results = self._search_reverse_acronym_finder(clue)
        
        if web_results:
            # Format the results nicely
            formatted_results = []
            for result in web_results[:5]:  # Limit to top 5
                formatted_results.append(f"{result} (from web)")
            return formatted_results
        
        # If no results from web either, try finding similar clues in dataset
        close_matches = []
        for existing_clue in self.clue_to_word_dict.keys():
            # Only consider clues with similar length to avoid irrelevant matches
            if 0.5 <= len(existing_clue) / len(clue) <= 2.0:
                # Calculate word overlap
                existing_words = set(re.findall(r'\b\w+\b', existing_clue.lower()))
                query_words = set(re.findall(r'\b\w+\b', clue.lower()))
                overlap = len(existing_words.intersection(query_words))
                
                if overlap >= 1:  # At least one word in common
                    close_matches.append((existing_clue, overlap))
        
        # Sort by overlap score
        close_matches.sort(key=lambda x: x[1], reverse=True)
        
        if close_matches:
            suggestions = []
            seen_acronyms = set()
            
            for i, (similar_clue, _) in enumerate(close_matches[:3]):  # Top 3
                for acronym in self.clue_to_word_dict[similar_clue]:
                    if acronym not in seen_acronyms:
                        suggestions.append(f"{acronym} (from: {similar_clue})")
                        seen_acronyms.add(acronym)
                        
                if len(suggestions) >= 3:  # Limit to 3 suggestions
                    break
            
            if suggestions:
                return [f"Similar clues found: {', '.join(suggestions)}"]
        
        return ["No acronyms found for this clue."]
    
    def search(self, query):
        """
        Search for either acronyms or clues depending on the query.
        
        Args:
            query (str): The acronym or clue to search for
            
        Returns:
            dict: Search results containing acronyms or clues
        """
        query = query.strip()
        
        # Check for question-format queries asking about acronym definitions
        question_patterns = [
            r"what(?:'s| is) the (?:full form|meaning|definition|expansion) of ([A-Z0-9]{2,7})\??",
            r"what does ([A-Z0-9]{2,7}) (?:stand for|mean)\??",
            r"define ([A-Z0-9]{2,7})",
            r"([A-Z0-9]{2,7}) (?:stands for|means|is short for)\??"
        ]
        
        # Check each pattern to see if it matches the query
        for pattern in question_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                # Extract the acronym from the question
                acronym = match.group(1).strip().upper()
                clues = self.get_clues_for_acronym(acronym)
                return {
                    'type': 'acronym',
                    'query': acronym,
                    'results': [{'clue': clue} for clue in clues]
                }
        
        # If query is short (2-7 chars) and all uppercase, likely an acronym
        if 2 <= len(query) <= 7 and query.isupper():
            clues = self.get_clues_for_acronym(query)
            return {
                'type': 'acronym',
                'query': query,
                'results': [{'clue': clue} for clue in clues]
            }
        else:
            # Treat as a clue
            acronyms = self.get_acronyms_for_clue(query)
            
            return {
                'type': 'clue',
                'query': query,
                'results': [{'acronym': acronym} for acronym in acronyms]
            }
        

    def _create_bidirectional_index(self):
        """Create an index of definitions to acronyms based on web scraping results."""
        self.definition_to_acronym = {}
        
        # Process cached web results to build the bidirectional index
        for acronym, definitions in self.web_cache.items():
            if isinstance(definitions, list):
                for definition in definitions:
                    # Remove the "(from web)" part if present
                    clean_def = definition.replace(" (from web)", "").strip().lower()
                    
                    if clean_def not in self.definition_to_acronym:
                        self.definition_to_acronym[clean_def] = []
                    
                    if acronym not in self.definition_to_acronym[clean_def]:
                        self.definition_to_acronym[clean_def].append(acronym)
        
        # Also add common acronyms
        for acronym, definitions in self.common_acronyms.items():
            for definition in definitions:
                clean_def = definition.lower()
                
                if clean_def not in self.definition_to_acronym:
                    self.definition_to_acronym[clean_def] = []
                
                if acronym not in self.definition_to_acronym[clean_def]:
                    self.definition_to_acronym[clean_def].append(acronym)


# Interactive demo with example
def run_interactive_demo():
    try:
        # Create an instance with the default path
        lookup = AcronymLookup()
        
        print("Advanced Acronym and Clue Lookup Demo")
        print("-------------------------------------")
        print("This version searches both local dataset and online sources!")
        print("Try searching for acronyms like 'NASA', 'HTML', 'RSVP'")
        print("Or try clues like 'Space agency', 'Hypertext Markup Language', etc.")
        print()
        
        while True:
            query = input("Enter search query (or 'quit' to exit): ")
            
            if query.lower() == 'quit':
                break
            
            if not query.strip():
                continue
            
            results = lookup.search(query)
            
            if results['type'] == 'acronym':
                print(f"\nClues for acronym '{results['query']}':")
                if results['results']:
                    for i, result in enumerate(results['results'], 1):
                        print(f"{i}. {result['clue']}")
                else:
                    print("No clues found for this acronym.")
            else:
                print(f"\nAcronyms for clue '{results['query']}':")
                if results['results']:
                    for i, result in enumerate(results['results'], 1):
                        print(f"{i}. {result['acronym']}")
                else:
                    print("No acronyms found for this clue.")
            
            print()
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Error details: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have the required libraries installed:")
        print("pip install pandas requests beautifulsoup4")


if __name__ == "__main__":
    # Check if required libraries are installed
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("Required libraries not found.")
        print("Please install them using:")
        print("pip install requests beautifulsoup4")
        sys.exit(1)
    
    # Run the interactive demo without requiring command-line arguments
    run_interactive_demo()