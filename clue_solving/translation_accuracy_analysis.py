import pandas as pd

def analyze_translation_accuracy(file_path):
    """
    Analyze the accuracy of the foreign language translation algorithm.
    Only prints the basic accuracy statistics.
    
    Args:
        file_path: Path to the CSV file containing translation results
        
    Returns:
        tuple: (total_clues, translatable, non_translatable, accuracy)
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Basic statistics
    total_clues = len(df)
    translatable = sum(df['TranslationStatus'] == True)
    non_translatable = sum(df['TranslationStatus'] == False)
    accuracy = (translatable / total_clues) * 100
    
    print(f"===== TRANSLATION ALGORITHM ACCURACY ANALYSIS =====")
    print(f"Total foreign language clues analyzed: {total_clues}")
    print(f"Clues that could be translated (True): {translatable}")
    print(f"Clues that could not be translated (False): {non_translatable}")
    print(f"Translation algorithm accuracy: {accuracy:.2f}%")
    
    return (total_clues, translatable, non_translatable, accuracy)

# Execute the analysis
if __name__ == "__main__":
    analyze_translation_accuracy("foreign_language_clues.csv")