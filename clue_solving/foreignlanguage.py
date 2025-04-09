import requests
import time
import random
import re
from urllib.parse import urlencode

######################################################################################################
# True Bidirectional Foreign Language Translation for Crosswords
######################################################################################################

def extract_translation_request(input_text):
    """
    Parse complex input strings with various patterns to extract:
    - what needs to be translated
    - source language
    - target language
    
    Supports true bidirectional translation between any language pair.
    
    Returns: (word_to_translate, source_lang, target_lang)
    """
    # Convert language names to language codes
    language_map = {
        'italian': 'it',
        'spanish': 'es',
        'french': 'fr',
        'german': 'de',
        'english': 'en',
        'italiano': 'it',
        'español': 'es',
        'espanol': 'es',
        'français': 'fr',
        'francais': 'fr',
        'deutsch': 'de',
        'inglés': 'en',
        'ingles': 'en'
    }
    
    # Define multiple patterns to match different types of clues
    
    # Pattern 1: "X, in Language" or "X in Language"
    pattern1 = r'([^,]+)(?:,)?\s+in\s+([A-Za-z]+)'
    
    # Pattern 2: "Language for X" or "Language word for X"
    pattern2 = r'([A-Za-z]+)(?:\s+word)?\s+for\s+(.+)'
    
    # Pattern 3: "X in Language" with various prepositions
    pattern3 = r'(.+)\s+(?:in|en|auf|in|a)\s+([A-Za-z]+)'
    
    # Pattern 4: "How to say X in Language"
    pattern4 = r'(?:how\s+to\s+say|como\s+se\s+dice)\s+(.+)\s+(?:in|en)\s+([A-Za-z]+)'
    
    # Pattern 5: "Translate X from LangA to LangB"
    pattern5 = r'(?:translate|traducir)\s+(.+)\s+from\s+([A-Za-z]+)\s+to\s+([A-Za-z]+)'
    
    # Pattern 6: "X from LangA to LangB"
    pattern6 = r'(.+)\s+from\s+([A-Za-z]+)\s+to\s+([A-Za-z]+)'
    
    # Pattern 7: "LangA X in LangB"
    pattern7 = r'([A-Za-z]+)\s+(.+)\s+in\s+([A-Za-z]+)'
    
    # Try pattern 5 (has explicit source and target)
    match = re.search(pattern5, input_text, re.IGNORECASE)
    if match:
        word = match.group(1).strip()
        source_language = match.group(2).strip().lower()
        target_language = match.group(3).strip().lower()
        source_lang = language_map.get(source_language)
        target_lang = language_map.get(target_language)
        if source_lang and target_lang:
            return (word, source_lang, target_lang)
    
    # Try pattern 6 (has explicit source and target)
    match = re.search(pattern6, input_text, re.IGNORECASE)
    if match:
        word = match.group(1).strip()
        source_language = match.group(2).strip().lower()
        target_language = match.group(3).strip().lower()
        source_lang = language_map.get(source_language)
        target_lang = language_map.get(target_language)
        if source_lang and target_lang:
            return (word, source_lang, target_lang)
    
    # Try pattern 7 (has explicit source and target)
    match = re.search(pattern7, input_text, re.IGNORECASE)
    if match:
        source_language = match.group(1).strip().lower()
        word = match.group(2).strip()
        target_language = match.group(3).strip().lower()
        source_lang = language_map.get(source_language)
        target_lang = language_map.get(target_language)
        if source_lang and target_lang:
            return (word, source_lang, target_lang)
    
    # Try pattern 1
    match = re.search(pattern1, input_text, re.IGNORECASE)
    if match:
        word = match.group(1).strip()
        language = match.group(2).strip().lower()
        target_lang = language_map.get(language)
        
        # Important: If target is English, we know the source is NOT English
        if target_lang == 'en':
            # Try to identify the specific non-English language
            source_lang = detect_language(word, exclude_english=True)
        else:
            # For other target languages, detect normally
            source_lang = detect_language(word)
            
        return (word, source_lang, target_lang)
    
    # Try pattern 2
    match = re.search(pattern2, input_text, re.IGNORECASE)
    if match:
        language = match.group(1).strip().lower()
        word = match.group(2).strip()
        target_lang = language_map.get(language)
        
        # Important: If target is non-English, source is likely English
        if target_lang and target_lang != 'en':
            source_lang = 'en'
        else:
            # Otherwise detect normally
            source_lang = detect_language(word)
            
        return (word, source_lang, target_lang)
    
    # Try pattern 3
    match = re.search(pattern3, input_text, re.IGNORECASE)
    if match:
        word = match.group(1).strip()
        language = match.group(2).strip().lower()
        target_lang = language_map.get(language)
        
        # Important: If target is English, we know the source is NOT English
        if target_lang == 'en':
            # Try to identify the specific non-English language
            source_lang = detect_language(word, exclude_english=True)
        else:
            # For other target languages, detect normally
            source_lang = detect_language(word)
            
        return (word, source_lang, target_lang)
    
    # Try pattern 4
    match = re.search(pattern4, input_text, re.IGNORECASE)
    if match:
        word = match.group(1).strip()
        language = match.group(2).strip().lower()
        target_lang = language_map.get(language)
        
        # Important: For "how to say" format, source is usually the language of the user
        # For English UI, assume English source unless obviously in another language
        source_lang = detect_language(word)
        
        return (word, source_lang, target_lang)
    
    # If no pattern matches or language not recognized
    return None

def detect_language(text, exclude_english=False):
    """
    Enhanced language detection based on character patterns, letter frequencies,
    and language-specific features without relying on word dictionaries.
    
    Args:
        text: Text to analyze
        exclude_english: If True, won't return English as the detected language
        
    Returns: 
        Most likely language code
    """
    # Clean and prepare the text
    text = text.lower().strip()
    
    # Language-specific character sets (strongest indicators)
    spanish_chars = ['ñ', 'á', 'é', 'í', 'ó', 'ú', '¿', '¡']
    french_chars = ['é', 'è', 'ê', 'à', 'â', 'ç', 'ô', 'ù', 'û', 'ï', 'œ']
    german_chars = ['ä', 'ö', 'ü', 'ß']
    italian_chars = ['à', 'è', 'ì', 'ò', 'ù']
    
    # Check for language-specific characters first (strongest indicator)
    for char in spanish_chars:
        if char in text:
            return 'es'
    for char in german_chars:
        if char in text:
            return 'de'
    for char in french_chars:
        if char in text:
            return 'fr'
    for char in italian_chars:
        if char in text:
            return 'it'
    
    # If we have "X in English" format, we need more advanced detection since X is non-English
    if exclude_english:
        # Analyze letter distribution and patterns for language detection
        
        # 1. Character frequency analysis
        letter_count = {}
        for char in text:
            if char.isalpha():
                letter_count[char] = letter_count.get(char, 0) + 1
        
        # Skip if the word is too short to analyze
        if len(text) >= 3:
            # Spanish language character patterns
            if ('ll' in text or 'rr' in text or 'ch' in text or text.endswith(('dad', 'ción', 'ar', 'er', 'ir'))):
                return 'es'
            
            # French language character patterns
            if (text.endswith(('eau', 'eux', 'oir', 'er', 'ez', 'ent')) or 
                'ou' in text or 'eu' in text or 'oi' in text or 'ph' in text):
                return 'fr'
            
            # Italian language character patterns
            if (text.endswith(('ino', 'one', 'are', 'ere', 'ire', 'zione')) or 
                'zz' in text or 'cch' in text or 'gli' in text or 'sc' in text):
                return 'it'
            
            # German language character patterns
            if (text.endswith(('ung', 'heit', 'keit', 'lich', 'ig', 'isch')) or 
                'sch' in text or 'tsch' in text or 'ck' in text or 'tz' in text):
                return 'de'
        
        # 2. Check consonant-vowel ratio (German has more consonants than Latin languages)
        vowels = sum(1 for c in text if c in 'aeiou')
        consonants = sum(1 for c in text if c.isalpha() and c not in 'aeiou')
        
        if len(text) >= 4:
            if consonants / (vowels + 0.001) > 2.0:  # High consonant ratio suggests German
                return 'de'
            
            # French often has more vowels
            if vowels / (len(text) + 0.001) > 0.55:
                return 'fr'
        
        # 3. Letter combinations and patterns
        # Spanish 
        if 'que' in text or 'qui' in text or text.endswith('o') or text.endswith('a'):
            return 'es'
        # French 
        if 'eau' in text or 'aux' in text or 'eux' in text or text.endswith('e'):
            return 'fr'
        # Italian
        if 'cce' in text or 'cci' in text or 'zza' in text or text.endswith('i'):
            return 'it'
        # German
        if 'cht' in text or 'tsch' in text or text.endswith('en'):
            return 'de'
            
        # Default for non-English detection (based on common language for crossword clues)
        if exclude_english:
            # Try to detect based on spelling patterns
            if any(text.endswith(s) for s in ('er', 'ez', 'eau', 'eux', 'ain', 'oir', 'eur')):
                return 'fr'  # French endings
            if any(text.endswith(s) for s in ('o', 'a', 'os', 'as', 'ar', 'ión')):
                return 'es'  # Spanish endings
            if any(text.endswith(s) for s in ('o', 'a', 'i', 'e', 'ino', 'one')):
                return 'it'  # Italian endings
            if any(text.endswith(s) for s in ('en', 'er', 'ung', 'heit', 'keit')):
                return 'de'  # German endings
                
            # If still undetermined, make an educated guess based on letter combinations
            if 'ch' in text or 'j' in text:
                return 'fr'  # French has many words with 'ch' and uses 'j'
                
            # Default to French as a common language for crossword clues
            return 'fr'
    
    # If no strong indicators and we don't need to exclude English, default to English
    return 'en'

# Removed levenshtein_distance function as it's no longer needed

def translate_word(word: str, source_lang: str, target_lang: str) -> str:
    """
    Direct translation of a word from source language to target language.
    
    Args:
        word: The word to translate
        source_lang: Language code (en, es, fr, it, de)
        target_lang: Language code (en, es, fr, it, de)
        
    Returns:
        Translated word
    """
    # If source and target are the same, no translation needed
    if source_lang == target_lang:
        return word
    
    # Try Google Translate (most reliable for direct translations)
    try:
        translation = google_translate_unofficial(word, source_lang, target_lang)
        if translation and translation.lower() != word.lower():
            print(f"Google translation: '{word}' → '{translation}'")
            return translation
    except Exception as e:
        print(f"Google Translate error: {str(e)}")
    
    # Fallback to MyMemory
    try:
        translation = mymemory_translate(word, source_lang, target_lang)
        if translation and translation.lower() != word.lower():
            print(f"MyMemory translation: '{word}' → '{translation}'")
            return translation
    except Exception as e:
        print(f"MyMemory API error: {str(e)}")
    
    # If all fails, return the original word
    print(f"Warning: Could not translate '{word}', returning original word")
    return word

def google_translate_unofficial(text: str, source_lang: str, target_lang: str) -> str:
    """
    Use the unofficial Google Translate API.
    """
    base_url = "https://translate.googleapis.com/translate_a/single"
    
    # Parameters for the request
    params = {
        "client": "gtx",
        "sl": source_lang,
        "tl": target_lang,
        "dt": "t",
        "q": text
    }
    
    # Add a small delay to prevent rate limiting
    time.sleep(random.uniform(0.2, 0.5))
    
    url = f"{base_url}?{urlencode(params)}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            translation = data[0][0][0] if data and len(data) > 0 and len(data[0]) > 0 else ""
            return translation
        else:
            print(f"Google Translate error: {response.status_code}")
            return ""
    except Exception as e:
        print(f"Google Translate error: {str(e)}")
        return ""

def mymemory_translate(text: str, source_lang: str, target_lang: str) -> str:
    """
    Use the MyMemory Translation API (free for limited usage).
    """
    # Combine language codes as required by the API
    lang_pair = f"{source_lang}|{target_lang}"
    
    url = "https://api.mymemory.translated.net/get"
    
    params = {
        "q": text,
        "langpair": lang_pair
    }
    
    # Add delay to prevent rate limiting
    time.sleep(random.uniform(0.2, 0.5))
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            translation = data.get("responseData", {}).get("translatedText", "")
            
            # Clean up the translation
            translation = translation.replace("&#39;", "'")
            
            return translation
        else:
            print(f"MyMemory API error: {response.status_code}")
            return ""
    except Exception as e:
        print(f"MyMemory API error: {str(e)}")
        return ""

def get_language_name(lang_code):
    """Convert language code to full language name"""
    language_names = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'it': 'Italian',
        'de': 'German'
    }
    return language_names.get(lang_code, lang_code)

######################################################################################################
# Main function
######################################################################################################

if __name__ == "__main__":
    print("===== Foreign Language Crossword Clue Processor =====")
    print("Available languages: English, Spanish, French, Italian, German")
    print("Examples of input:")
    print("- \"Eight, in Italian\"")
    print("- \"Spanish for dog\"")
    print("- \"How to say hello in French\"")
    print("- \"German word for house\"")
    print("- \"Translate hola from Spanish to English\"")
    print("- \"Bonjour from French to English\"")
    print("- \"Spanish gato in English\"")
    print("Or you can enter a word and select languages manually.")
    
    while True:
        clue_input = input("\nEnter clue (or 'exit' to quit): ")
        if clue_input.lower() == 'exit':
            break
        
        # Try to parse the input format first
        parsed_result = extract_translation_request(clue_input)
        
        if parsed_result:
            word, source_lang, target_lang = parsed_result
            source_name = get_language_name(source_lang)
            target_name = get_language_name(target_lang)
            print(f"\nDetected request to translate '{word}' from {source_name} to {target_name}")
        else:
            # If parsing fails, ask for manual input
            word = clue_input
            print("\nCouldn't detect language pattern. Please specify languages manually:")
            
            languages = {
                "1": "en", "2": "es", "3": "fr", "4": "it", "5": "de"
            }
            
            print("Source language:")
            print("1. English  2. Spanish  3. French  4. Italian  5. German")
            source_choice = input("Choose source language (1-5): ")
            source_lang = languages.get(source_choice, "en")
            
            print("\nTarget language:")
            print("1. English  2. Spanish  3. French  4. Italian  5. German")
            target_choice = input("Choose target language (1-5): ")
            target_lang = languages.get(target_choice, "es")
            
            source_name = get_language_name(source_lang)
            target_name = get_language_name(target_lang)
        
        # Perform translation
        print(f"\nTranslating: '{word}' from {source_name} to {target_name}")
        translated_answer = translate_word(word, source_lang, target_lang)
        
        print(f"\nTranslation: '{translated_answer}'")
        
        # Ask if the user wants to swap translation direction
        swap_option = input("\nWant to translate in the reverse direction? (y/n): ").lower()
        if swap_option == 'y':
            # Swap source and target languages
            print(f"\nTranslating: '{translated_answer}' from {target_name} to {source_name}")
            reverse_translation = translate_word(translated_answer, target_lang, source_lang)
            print(f"\nReverse translation: '{reverse_translation}'")
        
        # Continue or exit
        continue_option = input("\nTranslate another phrase? (y/n): ").lower()
        if continue_option != 'y':
            break
    
    print("\nThank you for using the Foreign Language Clue Processor!")