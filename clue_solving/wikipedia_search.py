import requests
import wikipediaapi
from collections import Counter
from nltk.corpus import wordnet
import nltk
import re
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import clue_classification_and_processing
from clue_classification_and_processing.clue_features import get_clues_dataframe 

# Load the dataset
df = get_clues_dataframe()

# Keep only relevant columns
df = df[['Clue', 'Word']].dropna()  # Drop missing values

# Display the first few rows
print(df.head())

# Load a pre-trained BERT model
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Ensure nltk dependencies are available
nltk.download('wordnet')
nltk.download('omw-1.4')

# Get a list of synonyms and related terms for a given word using WordNet. 
def get_synonyms(word):

    # Include the original word
    synonyms = set([word])  
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            # Replace underscores with spaces
            clean_word = lemma.name().replace("_", " ")  
            synonyms.add(clean_word)
    
    return list(synonyms)

# Expand words using BERT similarity search.
def get_bert_synonyms(word, top_n=10):
    
    candidate_words = get_synonyms(word)

    if not candidate_words:  # If no synonyms found, return just the original word
        return [word]

    # Compute BERT embeddings for input word
    word_embedding = bert_model.encode(word, convert_to_tensor=True)
    
    # Compute embeddings for candidate words
    candidate_embeddings = bert_model.encode(candidate_words, convert_to_tensor=True)

    # Compute cosine similarity between input word and candidate words
    similarities = util.pytorch_cos_sim(word_embedding, candidate_embeddings)

    # Option to return more or less results
    top_n = min(top_n, len(candidate_words))  
    if top_n == 0:
        # If no candidates are found, return the original word
        return [word]  
    
    # Get top N most similar words
    top_indices = torch.topk(similarities, top_n).indices.tolist()[0]
    
    # Return top similar words
    return [candidate_words[i] for i in top_indices]


# Extract words from the sentence and find their synonyms and related terms.
def extract_keywords(sentence):
    
    words = re.findall(r'\b\w+\b', sentence)  # Extract words
    expanded_keywords = set()
    for word in words:
        wordnet_synonyms = get_synonyms(word)  # WordNet synonyms
        bert_synonyms = get_bert_synonyms(word)  # BERT synonyms
        expanded_keywords.update(wordnet_synonyms + bert_synonyms)  # Merge both
    return list(expanded_keywords), words  # Return both expanded keywords and original words


 
# Search Wikipedia for articles containing the keywords.
def search_wikipedia_articles(keywords, original_words):
    
    wiki = wikipediaapi.Wikipedia(
        user_agent="Project/1.0 (contact: your-email@example.com)", 
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
                
                # Allow pages with at least one match
                if sum(word_matches) >= 1: 

                    # Store relevant pages
                    results[page_title] = page.text  

    else:
        print(f"Error fetching Wikipedia results for query: {search_query}")

    return results

# Merge two result dictionaries, avoiding duplicates
def merge_results(expanded_results, original_results):
    combined_results = {**expanded_results, **original_results} 
    return combined_results

# Count occurrences of keywords (words & synonyms) in Wikipedia text
def count_word_frequencies(text, keywords):
    
    # Tokenize Wikipedia text
    words = re.findall(r'\b\w+\b', text.lower())  
    word_counts = Counter(words)
    return {word: word_counts[word.lower()] for word in keywords if word.lower() in word_counts}

######################################################################################################
# Search Function
######################################################################################################

# Searches Wikipedia for articles related to the given crossword clue.
    #param sentence: The crossword clue to search for.
    #param word_length: (Optional) Exact word length to filter Wikipedia results.
    #return: A dictionary with Wikipedia search results and word frequency counts.
    
def search_wikipedia_for_clue(sentence, word_length=None):

    # Get words + synonyms and original words
    keywords, original_words = extract_keywords(sentence)  
    
    # Perform both searches
    expanded_results = search_wikipedia_articles(keywords, original_words)
    original_results = search_wikipedia_articles(original_words, original_words)

    # Merge results from both searches
    wiki_results = merge_results(expanded_results, original_results)

    word_frequencies = []
    # Store highest count for each word
    highest_counts = {}  
    # Store sum of counts for each Wikipedia page
    page_word_counts = {}  

    for title, content in wiki_results.items():
        # Count words & synonyms
        word_counts = count_word_frequencies(content, keywords)  
        
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
        # No filtering, return all results
        filtered_results = wiki_results  

    # Prepare output data
    result_data = {
        "highest_word_counts": highest_counts,
        "filtered_wikipedia_results": filtered_results,
        "word_frequencies": word_frequencies,
    }

    return result_data 

######################################################################################################
# Example Usage
######################################################################################################

clue = "Capital of France"
word_length = 5  # Optional, can be None
results = search_wikipedia_for_clue(clue, word_length)

# Print results
print("\nWikipedia Word Frequency Results:")
print("-" * 60)
for word, data in results["highest_word_counts"].items():
    print(f"{word.capitalize()} - Count: {data['count']} (Found in: {data['page']})")

print("\nAll Wikipedia pages retrieved:")
if results["filtered_wikipedia_results"]:
    for title in results["filtered_wikipedia_results"].keys():
        print(f"- {title}")
else:
    print("No Wikipedia pages matched the exact title length requirement.")


clue = "It's the freakin' weekend"
word_length = 5  # Optional, can be None
results = search_wikipedia_for_clue(clue, word_length)

# Print results
print("\nWikipedia Word Frequency Results:")
print("-" * 60)
for word, data in results["highest_word_counts"].items():
    print(f"{word.capitalize()} - Count: {data['count']} (Found in: {data['page']})")

print("\nAll Wikipedia pages retrieved:")
if results["filtered_wikipedia_results"]:
    for title in results["filtered_wikipedia_results"].keys():
        print(f"- {title}")
else:
    print("No Wikipedia pages matched the exact title length requirement.")

######################################################################################################
# Large Batch Testing
######################################################################################################

#Test Wikipedia search using a sample of clues from the dataset. 
# Checks if answers appear in Wikipedia **titles** AND/OR **page content**.
# Saves results to a CSV file.
# Used GenAI 
def test_wikipedia_search(sample_size=10, output_file="\testing_results\wikipedia_search_results.csv"):
    
    # Select a random sample of clues (avoids API rate limits)
    sample_df = df.sample(n=sample_size, random_state=42)

    results = []  # Store evaluation results

    for index, row in sample_df.iterrows(): 
        clue = row["clue"]
        answer = row["Word"].strip().lower() 

        print(f"\n ({index+1}/{sample_size}) Searching Wikipedia for: {clue} (Expected answer: {answer})")
        
        # Extract keywords using BERT and WordNet
        keywords, original_words = extract_keywords(clue)
        
        # Perform Wikipedia search
        expanded_results = search_wikipedia_articles(keywords, original_words)
        original_results = search_wikipedia_articles(original_words, original_words)

        # Merge results
        wiki_results = merge_results(expanded_results, original_results)

        # Initialize match conditions
        title_match = False
        content_match = False
        matched_page_title = "No Title Match"  # Default title if no match

        # Check if the answer appears in the Wikipedia page title or content
        for page_title, page_text in wiki_results.items():
            if answer in page_title.lower():  
                title_match = True  # Answer found in title
                matched_page_title = page_title  # Store the matching title
            
            if answer in page_text.lower():
                content_match = True  # Answer found in Wikipedia content
                
                # If no title match, update the matched page title with content match
                if not title_match:
                    matched_page_title = page_title  

            # Stop early if both conditions are met
            if title_match and content_match:
                break  
        
        # Store the result
        results.append({
            "clue": clue,
            "Expected Answer": answer,
            "Wikipedia Page Title": matched_page_title,  # Store title if found in content
            "Title Match": title_match,
            "Content Match": content_match
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate accuracy metrics
    title_accuracy = results_df["Title Match"].mean() * 100
    content_accuracy = results_df["Content Match"].mean() * 100

    print(f"\n✅ Wikipedia Title Match Accuracy: {title_accuracy:.2f}% ({results_df['Title Match'].sum()} out of {sample_size})")
    print(f"✅ Wikipedia Content Match Accuracy: {content_accuracy:.2f}% ({results_df['Content Match'].sum()} out of {sample_size})")

    # Save results to a CSV file
    results_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\n Results saved to: {output_file}")

# test_wikipedia_search(sample_size=20)

