"""
This file attempts to solve: given a search string, find the most likely wikipedia page on
that topic.
"""
import wikipediaapi

wiki = wikipediaapi.Wikipedia(user_agent='MyProjectName (merlin@example.com)', language='en')

example_page = wiki.page("John Smith")
category_string = ",".join(example_page.categories.keys())

def get_wikipedia_search_space(search_term):
    """

    :param search_term:
    :return:
    """

    # Given a search term, get a list of the best wikipedia pages


# If we've reached a disambiguation page, our search space should start as all the direct links
# (i.e. those which link to an article rather than another disambiguation page).
if 'article disambiguation pages' in category_string:
    print("This is a disambiguation page")
    print(f"Sample text:\n{example_page.text[0:500]}")

# Go through all the links in the example_page.links
for link in example_page.links:
    link_page = wiki.page(link)
    print(f"\n\nLink page='{link_page}'")
    print(f"Sample text:\n{link_page.text[0:500]}")
    input_break=input()


def print_links(page):
    links = page.links
    for title in sorted(links.keys()):
        print("%s: %s" % (title, links[title]))