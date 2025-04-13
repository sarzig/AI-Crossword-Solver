
import wikipedia

from clue_classification_and_processing.helpers import get_clues_by_class

# Optional: Set language (default is 'en')
wikipedia.set_lang("en")

# Your target name or last name
name = "Einstein"  # Replace with any last name

try:
    # Get the Wikipedia page for the name
    page = wikipedia.page(name)

    # Store the full page content in a variable
    page_text = page.content

    # Optionally, print the first few lines
    print(page_text[:500])  # Preview first 500 chars

except wikipedia.exceptions.DisambiguationError as e:
    print(f"Disambiguation page found for '{name}', suggestions: {e.options}")
    page_text = None

except wikipedia.exceptions.PageError:
    print(f"No page found for '{name}'")
    page_text = None


# xxx tbd accomplish this using name given profession reference.xlsx
b=get_clues_by_class(clue_class="Find name given profession/reference", classification_type="predicted_only", prediction_threshold=0.7)