import csv
import re

class RomanNumeralConverter:
    def __init__(self, csv_file):
        # Roman numeral mappings
        self.roman_map = {
            'I': 1, 'V': 5, 'X': 10, 'L': 50, 
            'C': 100, 'D': 500, 'M': 1000
        }
        
        # Number to Roman numeral mapping
        self.num_to_roman = [
            (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
            (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
            (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
        ]
        
        # Words for numbers
        self.num_to_word = {
            1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
            6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
            11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 
            15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen',
            19: 'nineteen', 20: 'twenty', 30: 'thirty', 40: 'forty',
            50: 'fifty', 60: 'sixty', 70: 'seventy', 80: 'eighty',
            90: 'ninety', 100: 'one hundred', 200: 'two hundred',
            300: 'three hundred', 400: 'four hundred', 500: 'five hundred',
            600: 'six hundred', 700: 'seven hundred', 800: 'eight hundred',
            900: 'nine hundred', 1000: 'one thousand', 2000: 'two thousand',
            3000: 'three thousand'
        }
        
        # Load the clues from CSV
        self.clues_dict = {}
        self.load_clues(csv_file)
        
        # Debug: Print the loaded clues
        print(f"Loaded {len(self.clues_dict)} clues from the CSV file")
        if len(self.clues_dict) > 0:
            print("Sample clues:")
            sample = list(self.clues_dict.items())[:3]
            for clue, answer in sample:
                print(f"  - Clue: '{clue}' -> Answer: '{answer}'")
    
    def load_clues(self, csv_file):
        """Load clues from the CSV file"""
        try:
            with open(csv_file, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Handle potential quotes in clues
                    clue = row['Clue'].strip('"\'')
                    self.clues_dict[clue.lower()] = row['Word']
        except FileNotFoundError:
            print(f"Warning: CSV file '{csv_file}' not found. Clue functionality will be limited.")
        except Exception as e:
            print(f"Error loading clues: {str(e)}")
    
    def roman_to_int(self, roman):
        """Convert a Roman numeral to an integer"""
        if not roman or not all(char in self.roman_map for char in roman):
            return None
            
        result = 0
        prev_value = 0
        
        for char in reversed(roman):
            current_value = self.roman_map[char]
            if current_value >= prev_value:
                result += current_value
            else:
                result -= current_value
            prev_value = current_value
            
        return result
    
    def int_to_roman(self, num):
        """Convert an integer to a Roman numeral"""
        if not isinstance(num, int) or num <= 0 or num > 3999:
            return None
            
        result = ''
        for value, numeral in self.num_to_roman:
            while num >= value:
                result += numeral
                num -= value
                
        return result
    
    def number_to_words(self, num):
        """Convert a number to words"""
        if num in self.num_to_word:
            return self.num_to_word[num]
            
        if num < 100:
            tens = (num // 10) * 10
            ones = num % 10
            if ones == 0:
                return self.num_to_word[tens]
            return f"{self.num_to_word[tens]}-{self.num_to_word[ones]}"
            
        if num < 1000:
            hundreds = (num // 100) * 100
            rest = num % 100
            if rest == 0:
                return self.num_to_word[hundreds]
            return f"{self.num_to_word[hundreds]} and {self.number_to_words(rest)}"
            
        thousands = (num // 1000) * 1000
        rest = num % 1000
        if rest == 0:
            return self.num_to_word[thousands]
        return f"{self.num_to_word[thousands]} {self.number_to_words(rest)}"
    
    def find_matching_clue(self, user_input):
        """Find a matching clue with more flexible comparison"""
        user_input = user_input.lower().strip()
        
        # Direct match
        if user_input in self.clues_dict:
            return self.clues_dict[user_input]
            
        # Try without punctuation
        clean_input = re.sub(r'[^\w\s]', '', user_input)
        for clue, answer in self.clues_dict.items():
            clean_clue = re.sub(r'[^\w\s]', '', clue)
            if clean_input == clean_clue:
                return answer
                
        # Check if the input is contained in any clue
        for clue, answer in self.clues_dict.items():
            if user_input in clue or clean_input in re.sub(r'[^\w\s]', '', clue):
                return answer
                
        return None
    
    def process_input(self, user_input):
        """Process user input: clue, number, or Roman numeral"""
        user_input = user_input.strip()
        
        # First check if it's a number - this should take priority
        if user_input.isdigit():
            num = int(user_input)
            roman = self.int_to_roman(num)
            if roman:
                return f"Roman numeral for {num}: {roman}"
            return "Please enter a number between 1 and 3999."
        
        # Then check if it's a Roman numeral
        if all(char in self.roman_map for char in user_input.upper()):
            roman = user_input.upper()
            num = self.roman_to_int(roman)
            if num:
                words = self.number_to_words(num)
                return f"Roman numeral {roman} represents: {words} ({num})"
            return "Invalid Roman numeral."
        
        # Finally check if it's a clue from the dataset
        matching_answer = self.find_matching_clue(user_input)
        if matching_answer:
            return f"Answer to clue: {matching_answer}"
                
        return "Input not recognized. Please enter a clue, number, or Roman numeral."

def main():
    # Get the CSV file path from user if needed
    csv_file = 'roman_output.csv'
    
    print("Roman Numeral Converter")
    print("Loading clues from:", csv_file)
    
    converter = RomanNumeralConverter(csv_file)
    
    print("\nEnter 'q' to quit")
    print("Examples:")
    print("- Enter '151, in old Rome' to get the answer from the clue")
    print("- Enter '42' to convert to Roman numerals")
    print("- Enter 'XIV' to convert from Roman numerals to words")
    
    while True:
        user_input = input("\nEnter a clue, number, or Roman numeral: ")
        if user_input.lower() == 'q':
            break
            
        result = converter.process_input(user_input)
        print(result)


if __name__ == "__main__":
    main()