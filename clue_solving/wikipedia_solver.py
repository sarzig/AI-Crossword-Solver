
import wikipedia

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