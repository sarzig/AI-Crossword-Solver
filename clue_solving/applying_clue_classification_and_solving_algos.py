"""
Author: Sarah

This file combines the two approaches of clue CLASSIFICATION and clue SOLVING.

It takes a dataframe as an input, which should have columns:
  * "clue"
  * "answer (optional column, for checking only)" OR "Word"

It applies predict_clues_df_from_default_pipeline and does keep_features=False,
which adds a column called "Top_Predicted_Classes" which is a list in this form:
[('Find name given profession/reference', 0.89), ('Foreign language', 0.04), ('Roman numeral', 0.04)]

It will solve any elements of "Top_Predicted_Classes" per the appropriate clue methodology
for that class.

Key functions:
* solve_clue(clue, clue_type, print_bool) - simple solver that takes a clue and a clue_type and attempts to solve
"""

from clue_classification_and_processing.clue_classification_machine_learning import \
    predict_clues_df_from_default_pipeline
from clue_classification_and_processing.helpers import get_project_root, print_if, process_text_into_clue_answer
from puzzle_objects.crossword_and_clue import get_crossword_from_csv
from web.nyt_html_to_standard_csv import get_random_clue_df_from_csv


def temporary_stupid_clue_solver(clue_text):
    # Stand-in just to make sure clue_type_to_methodology_lookup is working
    return "AHH" + clue_text + "AHH"


def solve_clue(clue_text, clue_type, clue_with_more_info=None, print_bool=False):
    """
    # xxx tbd - currently this has no allowance for already placed letters. I'm simply
    not sure that's in scope at this point :(

    Solves a crossword clue with methodology best suited for clue_type.

    If methodology exists for clue_type, applies it and returns the candidate answers
    after normalizing them into crossword answer format (i.e. tête -> TETE).

    If the methodology DNE, or methodology returns None, then None is passed.

    If the methodology returns a list of answers, all are normalized. No sorting
    is done, because it's assumed that some methodologies sort answers in list of
    likelihood.

    :param clue_text: clue text itself, no pre-processing
    :param clue_dictionary: optional parameter with richer clue information. Format is
    {"clue_text": , "clue_length", "clue_constraint": }

    :param clue_type: clue type, generally as determined by predict_clues_df_from_default_pipeline() (df) or
                      predict_single_clue_from_default_pipeline
    :param print_bool: if True, print helpful things about results and failures
    :return: a list of answers, normalized into crossword format, or None
    """

    # Step 1: Construct full clue_with_more_info if only clue_text is passed
    if clue_with_more_info is None:
        clue_with_more_info = {
            "clue_text": clue_text,
            "clue_length": None,
            "clue_constraint": None
        }
    else:
        # Ensure all keys are present in the dict, fill with fallback values if needed
        clue_with_more_info.setdefault("clue_text", clue_text)
        clue_with_more_info.setdefault("clue_length", None)
        clue_with_more_info.setdefault("clue_constraint", None)

    # Dictionary - this is THE algorithm to solve a clue of that class
    # This table shows the clue types that we *believe* we have the
    # ability to solve. If "None", then we currently have no way of solving it
    clue_type_to_methodology_lookup = {
        'Acronym/short form answer': None,
        'Analogy fill in the blank': None,
        'Basic fill in the blank':  temporary_stupid_clue_solver,
        'Biblical':  None,
        'Clever synonyms and examples': None,
        'Colloquial phrase fill in the blank': None,
        'Colloquial translation of phrase': None,
        'Direction answer': None,
        'Encyclopedia lookup': None,
        'Example of a class': None,
        'Find first name given last name': None,
        'Find name given profession/reference': None,
        'Foreign language': None,
        'Foreign language fill in the blank': None,
        'Full name of speaker given quote': None,
        'Members of a class': None,
        'One word synonym': None,
        'Proper noun fill in the blank': None,
        'Pun indicated by question mark': None,
        'Quote fill in the blank': None,
        'Quote phrase to word': None,
        'Roman numeral': None,
        'Short phrase colloquial synonym': None,
        'Straight definition': None,
        'Word before': None
    }

    # Find the appropriate solving methodology
    solve_methodology = clue_type_to_methodology_lookup.get(clue_type, None)

    # Exit function if solving methodology DNE
    if solve_methodology is None:
        print_if(statement=f"Clue '{clue_text}' was attempted to be solved with clue_type={clue_type} but no "
                           f"appropriate solving methodology exists for that clue type.",
                 print_bool=print_bool)
        return None

    # If there IS a solving methodology, apply it
    answer = solve_methodology(clue_with_more_info)

    # If answer is None, return None
    if answer is None:
        print_if(statement=f"Clue '{clue_text}' was attempted to be solved with clue_type={clue_type}, "
                           f"and the methodology {solve_methodology} was attempted, however there was no return.",
                 print_bool=print_bool)
        return None

    # If answer is a string, then set candidate_answers = [answer]
    candidate_answers = []
    if isinstance(answer, str):
        candidate_answers.append(answer)  # Convert string to a list

    # If answer is a list, leave as it
    elif isinstance(answer, list):
        candidate_answers = answer

    # If there is a result, normalize it
    for i in range(len(candidate_answers)):
        candidate_answers[i] = process_text_into_clue_answer(candidate_answers[i])
        process_text_into_clue_answer(answer)

    return candidate_answers

''' Workspace '''

csv_path = f"{get_project_root()}/data/puzzle_samples/processed_puzzle_samples/crossword_2022_07_24.csv"

crossword = get_crossword_from_csv(csv_path)
all_clues_predicted = predict_clues_df_from_default_pipeline(clues_df=crossword.clue_df,
                                                             keep_features=False,
                                                             top_n=3)

# Filter the rows where the 'Top_Predicted_Classes' contains a class with a probability > 0.5
filtered_clues = all_clues_predicted[
    all_clues_predicted['Top_Predicted_Classes'].apply(
        lambda x: any(prob > 0.8 for _, prob in x)  # Check if any probability in the dictionary is greater than 0.5
    )
]

filtered_clues.to_csv("crossword_2022_07_24_with_high_likelihood.csv")
filtered_clues.to_csv("crossword_2022_07_24_ENTIRE_ENRICHED_DF.csv")

# Print the filtered dataframe
print(filtered_clues)


'''
input
dataframe

applies predicted classes
only attempts to solve those classes with likelihood over 0.8
returns dictionary with added columns: Answers

Pipeline proposed:
* start with section with most SOLVED


WHAT CAN I GIVE SHERYL THAT'S USEFUL
number_direction
answer_certainty
answer_list
'''