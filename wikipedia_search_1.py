import requests
import wikipediaapi
from collections import Counter
from nltk.corpus import wordnet
import nltk
import re
import os

# Ensure nltk dependencies are available
nltk.download('wordnet')
nltk.download('omw-1.4')

def get_synonyms(word):
    """ Get a list of synonyms and related terms for a given word using WordNet. """
    synonyms = set([word])  # Include the original word
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            clean_word = lemma.name().replace("_", " ")  # Replace underscores with spaces
            synonyms.add(clean_word)
    
    return list(synonyms)

def extract_keywords(sentence):
    """ Extract words from the sentence and find their synonyms and related terms. """
    words = re.findall(r'\b\w+\b', sentence)  # Extract words
    expanded_keywords = set()
    for word in words:
        expanded_keywords.update(get_synonyms(word))  # Add both words and synonyms
    return list(expanded_keywords), words  # Return both expanded keywords and original words

def search_wikipedia_articles(keywords, original_words):
    """ 
    Search Wikipedia for articles containing the keywords but restrict final titles if needed.
    """
    wiki = wikipediaapi.Wikipedia(
        user_agent="Project/1.0 (contact: your-email@example.com)",  # Replace with your details
        language="en"
    )
    results = {}

    # Use multiple keywords together for better search results
    search_query = " ".join(keywords[:10])  # Use up to 10 keywords for broader search

    search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={search_query}&srlimit=50&format=json"
    response = requests.get(search_url, headers={"User-Agent": "MindStormBot/1.0 (contact: your-email@example.com)"})
    
    if response.status_code == 200:
        data = response.json()
        search_results = data.get("query", {}).get("search", [])
        for entry in search_results:
            page_title = entry["title"]

            # Exclude pages where the title is an EXACT match to an original input word
            if any(word.lower() == page_title.lower() for word in original_words):
                continue

            page = wiki.page(page_title)
            if page.exists():
                page_text = page.text.lower()

                # Ensure at least one original word or synonym appears
                word_matches = [word.lower() in page_text for word in original_words]
                if sum(word_matches) >= 1:  # Allow pages with at least one match
                    results[page_title] = page.text  # Store relevant pages

    else:
        print(f"Error fetching Wikipedia results for query: {search_query}")

    return results

def count_word_frequencies(text, keywords):
    """ Count occurrences of keywords (words & synonyms) in Wikipedia text. """
    words = re.findall(r'\b\w+\b', text.lower())  # Tokenize Wikipedia text
    word_counts = Counter(words)
    return {word: word_counts[word.lower()] for word in keywords if word.lower() in word_counts}

def main():
    sentence = input("Enter a sentence (clue): ")
    word_length_input = input("Enter the exact word length for Wikipedia results (or press Enter to show all): ").strip()
    
    # If word length is provided, convert it to an integer; otherwise, set to None
    word_length = int(word_length_input) if word_length_input.isdigit() else None

    keywords, original_words = extract_keywords(sentence)  # Get words + synonyms and original words
    wiki_results = search_wikipedia_articles(keywords, original_words)  # Search Wikipedia

    # If no results found, retry with original words only (no synonyms)
    if not wiki_results:
        print("No results found with expanded keywords. Retrying with original words only...")
        wiki_results = search_wikipedia_articles(original_words, original_words)

    word_frequencies = []
    highest_counts = {}  # Stores highest count for each word
    page_word_counts = {}  # Stores sum of counts for each Wikipedia page

    for title, content in wiki_results.items():
        word_counts = count_word_frequencies(content, keywords)  # Count words & synonyms
        
        # Track highest count for each word
        for word, count in word_counts.items():
            if word not in highest_counts or count > highest_counts[word]['count']:
                highest_counts[word] = {"count": count, "page": title}
        
        # Calculate total mentions of all words in each Wikipedia page
        total_count = sum(word_counts.values())
        page_word_counts[title] = total_count

        for word, count in word_counts.items():
            word_frequencies.append({"Wikipedia Page": title, "Word": word, "Count": count})
    
    # If word length is provided, filter Wikipedia results based on exact title length
    if word_length:
        filtered_results = {title: text for title, text in wiki_results.items() if len(title) == word_length}
    else:
        filtered_results = wiki_results  # No filtering, return all results

    # Print results
    print("\nWikipedia Word Frequency Results:")
    print("-" * 60)

    # Print highest count for each word
    for word, data in highest_counts.items():
        print(f"{word.capitalize()} - Count: {data['count']} (Found in: {data['page']})")

    print("\nAll Wikipedia pages retrieved:")
    if filtered_results:
        for title in filtered_results.keys():
            print(f"- {title}")
    else:
        print("No Wikipedia pages matched the exact title length requirement.")

if __name__ == "__main__":
    main()