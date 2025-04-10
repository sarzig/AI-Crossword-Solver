import pandas as pd
import requests
import time
import random
import re
from urllib.parse import urlencode

import sys
import os

# Add the parent directory to the module search path so that we can import project files
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from clue_classification_and_processing.helpers import get_clues_by_class

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

    # Dictionary of common direction indicators for translation
    language_direction = {
        'english': {'to': 'en', 'from': set(['es', 'fr', 'it', 'de'])},
        'spanish': {'to': 'es', 'from': set(['en', 'fr', 'it', 'de'])},
        'french': {'to': 'fr', 'from': set(['en', 'es', 'it', 'de'])},
        'italian': {'to': 'it', 'from': set(['en', 'es', 'fr', 'de'])},
        'german': {'to': 'de', 'from': set(['en', 'es', 'fr', 'it'])}
    }

    # City/country to language mapping
    location_to_language = {
        'paris': 'fr',
        'france': 'fr',
        'spain': 'es',
        'madrid': 'es',
        'barcelona': 'es',
        'italy': 'it',
        'rome': 'it',
        'milan': 'it',
        'germany': 'de',
        'berlin': 'de',
        'munich': 'de'
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

    # Pattern 8: "Language Word" - Simple language adjective with word (e.g., "Spanish water", "French romance")
    pattern8 = r'^([A-Za-z]+)\s+([A-Za-z]+)$'

    # Pattern 9: "City Word" - City name followed by a word (e.g., "Paris pal")
    pattern9 = r'^([A-Za-z]+)\s+([A-Za-z]+)$'

    # Pattern 10: "Word, in City" - Word followed by a city (implying language)
    pattern10 = r'([^,]+)(?:,)?\s+in\s+([A-Za-z]+)'

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

    # Try pattern 8 (Language Word pattern, e.g., "Spanish water", "French romance")
    match = re.search(pattern8, input_text, re.IGNORECASE)
    if match:
        language = match.group(1).strip().lower()
        word = match.group(2).strip()

        # Check if the first word is a language
        if language in language_map:
            target_lang = language_map[language]
            source_lang = 'en'  # Assume English as source for these common crossword formats
            return (word, source_lang, target_lang)

    # Try pattern 9 (City Word pattern, e.g., "Paris pal")
    match = re.search(pattern9, input_text, re.IGNORECASE)
    if match:
        location = match.group(1).strip().lower()
        word = match.group(2).strip()

        # Check if the first word is a city/country that implies a language
        if location in location_to_language:
            target_lang = location_to_language[location]
            source_lang = 'en'  # Assume English as source for these common crossword formats
            return (word, source_lang, target_lang)

    # Try pattern 10 (Word, in City pattern, e.g., "Fine, in Paris")
    match = re.search(pattern10, input_text, re.IGNORECASE)
    if match:
        word = match.group(1).strip()
        location = match.group(2).strip().lower()

        # Check if the second part is a city/country that implies a language
        if location in location_to_language:
            target_lang = location_to_language[location]
            source_lang = 'en'  # Assume English as source for these common crossword formats
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
            # For other target languages, source is usually English in crossword contexts
            # This helps with ambiguous cases like "Chat in Spanish" vs "Chat in English"
            source_lang = 'en'

            # But if the word clearly contains non-English characters, detect appropriately
            if any(c in word for c in 'áéíóúñüçèêàòù'):
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

    # Special case handling for direct language references without clear patterns
    # For crossword clues like "Spanish water", "French romance"
    words = input_text.strip().split()
    if len(words) >= 2:
        first_word = words[0].lower()
        rest_of_phrase = ' '.join(words[1:])

        # Check if first word is a language
        if first_word in language_map:
            target_lang = language_map[first_word]
            source_lang = 'en'  # Assume English as source language
            return (rest_of_phrase, source_lang, target_lang)

        # Check if first word is a city/country that implies a language
        if first_word in location_to_language:
            target_lang = location_to_language[first_word]
            source_lang = 'en'  # Assume English as source language
            return (rest_of_phrase, source_lang, target_lang)

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

    # Special case handling for crossword clues with "in English" pattern
    # For "X in English", we're translating TO English, and X should be a foreign word
    if target_lang == 'en' and word.lower() == word and all(c.isalpha() or c.isspace() for c in word):
        # Try all possible source languages if not specified or if source is ambiguous
        possible_sources = ['es', 'fr', 'it', 'de']
        if source_lang != 'en':
            # If source is already specified and not English, try that first
            possible_sources = [source_lang] + [lang for lang in possible_sources if lang != source_lang]

        # Try each potential source language
        for potential_source in possible_sources:
            try:
                translation = google_translate_unofficial(word, potential_source, 'en')
                if translation and translation.lower() != word.lower():
                    print(f"Found translation from {potential_source} to en: '{word}' → '{translation}'")
                    return translation
            except Exception:
                continue

    # For "Word in Language" where Language is not English (more common in crosswords)
    # Default from English if there's no special characters
    if target_lang != 'en' and source_lang == 'en':
        try:
            translation = google_translate_unofficial(word, 'en', target_lang)
            if translation and translation.lower() != word.lower():
                return translation
        except Exception as e:
            print(f"Google Translate error: {str(e)}")

    # Try with the specified source and target
    try:
        translation = google_translate_unofficial(word, source_lang, target_lang)
        if translation and translation.lower() != word.lower():
            return translation
    except Exception as e:
        print(f"Google Translate error: {str(e)}")

    # If translating to English still fails, try reversing direction as a last resort
    if target_lang == 'en':
        # Try some common foreign words that might be mistaken for English
        common_foreign_words = {
            'chat': 'fr',  # cat in French
            'die': 'de',  # the in German
            'sea': 'es',  # be in Spanish
            'red': 'es',  # network in Spanish
            'si': 'it',  # yes in Italian
            'no': 'it',  # no in Italian
            'casa': 'es',  # house in Spanish
            'pan': 'es',  # bread in Spanish
            'que': 'fr',  # what/that in French
            'me': 'es',  # me in Spanish
            'via': 'it',  # way in Italian
            'mal': 'fr',  # bad in French
        }

        word_lower = word.lower()
        if word_lower in common_foreign_words:
            potential_source = common_foreign_words[word_lower]
            try:
                translation = google_translate_unofficial(word_lower, potential_source, 'en')
                if translation and translation.lower() != word.lower():
                    return translation
            except Exception:
                pass

    # Try all source languages for translating TO English as a final attempt
    if target_lang == 'en':
        all_sources = ['fr', 'es', 'it', 'de']
        for try_source in all_sources:
            if try_source != source_lang:  # Skip the one we already tried
                try:
                    translation = google_translate_unofficial(word, try_source, 'en')
                    if translation and translation.lower() != word.lower():
                        return translation
                except Exception:
                    continue

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


def solve_foreign_language(clue):
    """
    Process a foreign language crossword clue and return possible answers.

    Args:
        clue: The crossword clue text (e.g., "Eight, in Italian", "Spanish for dog")

    Returns:
        A single answer string or a list of possible answers
    """
    # Try to parse the clue
    parsed_result = extract_translation_request(clue)

    if not parsed_result:
        return ["Could not parse foreign language clue"]

    word, source_lang, target_lang = parsed_result

    # Perform the translation
    translated_answer = translate_word(word, source_lang, target_lang)

    # If we have a clear translation, return it as a single answer
    if translated_answer and translated_answer.lower() != word.lower():
        return translated_answer

    # Otherwise return a list with the translated answer and a backup message
    return [translated_answer, "Translation uncertain"]


######################################################################################################
# Main function
######################################################################################################

if __name__ == "__main__":
    while True:

        foreign_language_clues = get_clues_by_class(clue_class="Foreign language", classification_type="manual_only")

        # print("foreign_language_clues: \n", foreign_language_clues)

        # Create a DataFrame from the clues
        df = pd.DataFrame(foreign_language_clues, columns=["ID", "Clue", "Word", "Class"])

        output_filename = "foreign_language_clues.csv"

        # Save to CSV file in the same directory as the script
        df.to_csv(output_filename, index=False)

        print(f"Foreign language clues saved to {output_filename}")

        clue_input = input("\nEnter clue (or 'exit' to quit): ")
        if clue_input.lower() == 'exit':
            break

        # Parse the clue and translate
        parsed_result = extract_translation_request(clue_input)
        if parsed_result:
            word, source_lang, target_lang = parsed_result
            translated_answer = translate_word(word, source_lang, target_lang)
            print(translated_answer)  # Output only the translation
        else:
            print("Error")  # Only display "Error" if the clue could not be parsed

        # No additional text or prompts after showing the translation
        continue  # Moves directly to the next iteration to accept a new clue