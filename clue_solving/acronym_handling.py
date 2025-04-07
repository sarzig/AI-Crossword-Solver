import requests
import re
from bs4 import BeautifulSoup
import time

class AcronymFinder:
    def __init__(self):
        # Empty caches for storing results during the session
        self.acronym_cache = {}
        self.phrase_cache = {}
    
    def find_all_meanings(self, acronym):
        """Find all possible meanings for an acronym"""
        if not acronym:
            return ["Please enter an acronym"]
        
        # Clean up the acronym
        acronym = acronym.strip().upper().replace('.', '')
        
        # Check cache first for faster responses
        if acronym in self.acronym_cache:
            return self.acronym_cache[acronym]
        
        # Collect results from multiple sources
        all_meanings = []
        
        # Source 1: Acromine API
        acromine_results = self._search_acromine(acronym)
        all_meanings.extend(acromine_results)
        
        # Source 2: Abbreviations.com
        abbreviations_results = self._search_abbreviations_com(acronym)
        for result in abbreviations_results:
            if result not in all_meanings:
                all_meanings.append(result)
        
        # Source 3: Acronym Finder
        acronymfinder_results = self._search_acronymfinder(acronym)
        for result in acronymfinder_results:
            if result not in all_meanings:
                all_meanings.append(result)
        
        # Cache results for future lookups
        if all_meanings:
            self.acronym_cache[acronym] = all_meanings
            return all_meanings
        else:
            return ["No meanings found for this acronym"]
    
    def generate_acronyms(self, phrase):
        """Generate possible acronyms for a phrase"""
        if not phrase:
            return ["Please enter a phrase"]
        
        phrase = phrase.strip()
        
        # Check cache
        if phrase.upper() in self.phrase_cache:
            return self.phrase_cache[phrase.upper()]
        
        # Generate possible acronyms
        acronyms = []
        
        # Basic acronym (first letter of each word)
        words = phrase.split()
        if len(words) > 1:
            basic_acronym = ''.join([word[0].upper() for word in words if word])
            acronyms.append(basic_acronym)
        
        # Acronym without small words
        important_words = [word for word in words if word.lower() not in ['of', 'the', 'and', 'in', 'for', 'to', 'a', 'an']]
        if len(important_words) > 1:
            important_acronym = ''.join([word[0].upper() for word in important_words])
            if important_acronym not in acronyms:
                acronyms.append(important_acronym)
        
        # Acronym using first letter of first word and first letters of important words in rest
        if len(words) > 2:
            first_word = words[0]
            rest_important = [word for word in words[1:] if word.lower() not in ['of', 'the', 'and', 'in', 'for', 'to', 'a', 'an']]
            if first_word and rest_important:
                mixed_acronym = first_word[0].upper() + ''.join([word[0].upper() for word in rest_important])
                if mixed_acronym not in acronyms:
                    acronyms.append(mixed_acronym)
        
        # Cache and return results
        if acronyms:
            self.phrase_cache[phrase.upper()] = acronyms
            return acronyms
        else:
            return ["Could not generate acronym for this phrase"]
    
    def _search_acromine(self, acronym):
        """Search for acronym meanings using the Acromine API"""
        try:
            url = f"http://www.nactem.ac.uk/software/acromine/dictionary.py?sf={acronym}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0 and 'lfs' in data[0]:
                    return [item['lf'] for item in data[0]['lfs']]
            return []
        except Exception as e:
            print(f"Acromine API error: {e}")
            return []
    
    def _search_abbreviations_com(self, acronym):
        """Search for acronym meanings on Abbreviations.com"""
        try:
            url = f"https://www.abbreviations.com/{acronym}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                meanings = []
                
                # Extract meanings from relevant HTML elements
                for desc in soup.select('div.desc'):
                    meaning = desc.get_text(strip=True)
                    if meaning and meaning != acronym:
                        meanings.append(meaning)
                
                return meanings
            return []
        except Exception as e:
            print(f"Abbreviations.com search error: {e}")
            return []
    
    def _search_acronymfinder(self, acronym):
        """Search for acronym meanings on AcronymFinder.com"""
        try:
            url = f"https://www.acronymfinder.com/{acronym}.html"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                meanings = []
                
                # Extract meanings from the table
                tables = soup.select('table.table-striped')
                if tables:
                    for row in tables[0].select('tr'):
                        cells = row.select('td')
                        if len(cells) >= 2:
                            meaning = cells[1].get_text(strip=True)
                            if meaning:
                                meanings.append(meaning)
                
                return meanings
            return []
        except Exception as e:
            print(f"AcronymFinder.com search error: {e}")
            return []

def main():
    finder = AcronymFinder()
    
    print("Acronym Lookup Tool")
    print("==================")
    print("Type an acronym to find all possible meanings")
    print("Type a phrase to generate possible acronyms")
    print("Type 'q' to quit\n")
    
    while True:
        user_input = input("Enter text: ").strip()
        
        if user_input.lower() in ['q', 'quit', 'exit']:
            break
        
        if not user_input:
            continue
        
        # Determine if input is likely an acronym or a phrase
        if user_input.upper() == user_input and len(user_input) >= 2 and len(user_input) <= 15 and ' ' not in user_input:
            # Process as acronym
            print(f"\nSearching for all meanings of {user_input.upper()}...")
            results = finder.find_all_meanings(user_input)
            
            print(f"\nMeanings of {user_input.upper()}:")
            if len(results) == 1 and results[0].startswith("No meanings"):
                print(results[0])
            else:
                for i, meaning in enumerate(results):
                    print(f"{i+1}. {meaning}")
        else:
            # Process as phrase
            print(f"\nGenerating acronyms for '{user_input}'...")
            results = finder.generate_acronyms(user_input)
            
            print(f"\nPossible acronyms for '{user_input}':")
            if len(results) == 1 and results[0].startswith("Could not"):
                print(results[0])
            else:
                for i, acronym in enumerate(results):
                    print(f"{i+1}. {acronym}")
        
        print()  # Add blank line for readability

if __name__ == "__main__":
    main()