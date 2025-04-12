"""
Author: Sarah

This file attempts to solve: given a search string, find the most likely wikipedia page on
that topic.

NOTES: xxx tbd this is half-baked, and I never quite got this to run fast enough to be acceptable.
"""

import wikipediaapi
from clue_classification_and_processing.helpers import print_if

wiki = wikipediaapi.Wikipedia(user_agent='MyProjectName (merlin@example.com)', language='en')

example_page = wiki.page("John Smith")
category_string = ",".join(example_page.categories.keys())


def is_answer_in_wikipedia_search_space(search_term, answer, print_statement=True):
    """
    Given a search term and an answer, this invokes get_wikipedia_search_space()
    to (1) find the search space, (2) search every page in the search space, and
    (3) return a dictionary with the:
        - page_name where match occurred
        - number of matches
        - list of all match context (match +- 200 characters)

    :param answer: the answer to check for within encyclopedia lookup of search_term
    :param search_term: term to search in wikipedia
    :param print_statement: whether to print
    :return: False if not found, otherwise return the dictionary of the answers
    """
    search_space = get_wikipedia_search_space(search_term)

    # If nothing is found using that term, return False
    if search_space is None:
        return False


def get_wikipedia_search_space(search_term, print_statement=True):
    """
    Given a search term, get a list of the wikipedia page or pages that have that search term.

    Examples:
    * search_term="Malcolm" pings a disambiguation page and would return a list of
      all non-disambiguation pages on the first disambiguation page
    * search_Term="Malcolm X" directly gives the page, so it returns ["Malcolm X"].
                  In this case, "Malcolm X" is the official title for the page, and
                  wikipediaapi.search(Malcolm X) will yield the correct page
    * search_term="Malcolm Little" and "el-Hajj Malik el-Shabazz" also points to the
                  "Malcolm X" page (This logic is done within Wikipedia, not within MY work).
                  However, the official page title is "Malcolm X",
                  and get_wikipedia_search_space returns ["Malcolm X"].

    :param print_statement: whether to print
    :param search_term: term to search in wikipedia
    :return: list of wikipedia page links. A wikipedia link is a plain text, exact term with
             which you can call page = wiki.page(wiki_link)
    """

    # Attempt to get the page
    page_attempt = wiki.page(search_term)
    category_string = ",".join(page_attempt.categories.keys())

    # A failed page attempt still yields a page, albeit one with no text.
    # check for no text on page
    if page_attempt.text == "":
        print_if(f"Search term '{search_term}' did not yield a match", print_statement)
        return None

    # If page is a disambiguation page, we need to generate a list of links
    # we exclude additional disambiguation pages, unless there are NO pages that
    # are non disambiguation pages
    if 'article disambiguation pages' in category_string:
        all_links = []
        for link in page_attempt.links:
            if "disambiguation" not in link.lower():
                all_links.append(link)
        if len(all_links) == 0:
            all_links = page_attempt.links

        return all_links

    # If page is a direct hit, return that page
    else:
        return [page_attempt.title]


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
