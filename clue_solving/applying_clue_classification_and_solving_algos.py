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
* solve_clue(clue_text, clue_type, clue_with_more_info, print_bool=False) - Main function!
* return_methodology_by_clue_type(clue_type)
* return_constraint_hits(options, clue_with_more_info)

Wrappers and helpers for those wrappers
* synonym_wrapper(clue_with_more_info) - Solves clues classified as "One word synonym" using recursive synonym search
    * stem_recursive_synonyms(recursive_synonyms)
* foreign_language_wrapper(clue_with_more_info)
* most_common_solver(clue_with_more_info)






"""
import os
import re
from itertools import permutations

import pandas as pd
from clue_classification_and_processing.clue_classification_machine_learning import \
    predict_clues_df_from_default_pipeline
from clue_classification_and_processing.helpers import get_project_root, print_if, process_text_into_clue_answer, \
    get_most_common_clue_word_pair
from clue_solving.foreign_language import solve_foreign_language
from clue_solving.synonym_search import get_recursive_synonyms
from puzzle_objects.crossword_and_clue import get_crossword_from_csv
from web.nyt_html_to_standard_csv import get_random_clue_df_from_csv


def return_methodology_by_clue_type(clue_type):
    """
    Given a clue_type, returns the solving methodology for that clue type.

    Every methodology mentioned should take clue_with_more_info as the ONLY
    argument!

    :param clue_type: clue type
    :return: solving methodology, or None
    """
    # xxx tbd ADD new clue solving methods!
    clue_type_to_methodology_lookup = {
        'Acronym/short form answer': None,
        'Analogy fill in the blank': None,
        'Basic fill in the blank': None,
        'Biblical': None,
        'Clever synonyms and examples': None,
        'Colloquial phrase fill in the blank': None,
        'Colloquial translation of phrase': None,
        'Direction answer': None,
        'Encyclopedia lookup': None,
        'Example of a class': None,
        'Find first name given last name': None,
        'Find name given profession/reference': None,
        'Foreign language': foreign_language_wrapper,
        'Foreign language fill in the blank': None,
        'Full name of speaker given quote': None,
        'Members of a class': None,
        'Most common crossword clues': most_common_solver,
        'One word synonym': synonym_wrapper,
        'Proper noun fill in the blank': None,
        'Pun indicated by question mark': None,
        'Quote fill in the blank': None,
        'Quote phrase to word': None,
        'Roman numeral': roman_numeral_possible_solutions,
        'Short phrase colloquial synonym': None,
        'Straight definition': None,
        'Word before': None,
    }

    # Find the appropriate solving methodology or None if None exists
    return clue_type_to_methodology_lookup.get(clue_type, None)


def return_constraint_hits(options, clue_with_more_info):
    """
    Given a list of options and a constraint like "..a..", return all clues which
    meet that. If constraint is none, then generate a constraint thing based off
    the answer length

    :param options: options - either a single string, or a list of strings
    :param clue_with_more_info: a clue_with_more_info dictionary
    :return: a pared down list of options that meet that constraint, or None
    """

    # If it's just one thing, give that
    if isinstance(options, str):
        options = [options]  # convert string to a list

    # convert options to crossword format
    options = [process_text_into_clue_answer(x) for x in options]

    # Build regex pattern from constraint
    # If the constraint is none, then just use the length of the clue and all dots
    constraint = clue_with_more_info["clue_constraint"]
    answer_length = clue_with_more_info["answer_length"]
    if constraint is None:
        if answer_length is None:
            return ["ERROR in return_constraint_hits"]
        else:
            constraint = "." * clue_with_more_info["answer_length"]

            pattern = "^" + constraint + "$"
            regex = re.compile(pattern, re.IGNORECASE)

            return [word for word in options if regex.match(word)]


def synonym_wrapper(clue_with_more_info):
    """
    Wraps Sheryl's "get_recursive_synonyms" - hard codes depth=3.

    xxx tbd could do depth=5 but limit it based on execution TIME.

    :param clue_with_more_info: a dictionary like {"clue_text":, "answer_length":, "clue_constraint": }
    :return: a list of valid options, properly formatted
    """
    clue_text = clue_with_more_info["clue_text"]
    answer_length = clue_with_more_info["answer_length"]
    clue_constraint = clue_with_more_info["clue_constraint"]
    print(f"clue_with_more_info={clue_with_more_info}")

    print(f"Clue_text={clue_text}")

    # Get list of recursive synonyms
    recursive_synonyms = get_recursive_synonyms(clue_text, depth=3)
    recursive_synonyms_including_stems = stem_recursive_synonyms(recursive_synonyms)

    # Get list of options that meet the given constraints
    valid_options = return_constraint_hits(options=recursive_synonyms_including_stems,
                                           clue_with_more_info=clue_with_more_info)

    return valid_options


def foreign_language_wrapper(clue_with_more_info):
    """
    This wraps Swathi/Eroniction's solve_foreign_language

    :param clue_with_more_info: a dictionary like {"clue_text": "", "answer_length": None, "clue_constraint": None}
    :return: a list of valid options, properly formatted
    """

    clue_text = clue_with_more_info["clue_text"]
    answer_length = clue_with_more_info["answer_length"]
    clue_constraint = clue_with_more_info["clue_constraint"]

    # If the result is a string, return whatever return_constraint_hits gives!
    possible_answer = solve_foreign_language(clue_text)
    if isinstance(possible_answer, str):
        return return_constraint_hits(possible_answer,
                                      clue_with_more_info=clue_with_more_info)


def most_common_solver(clue_with_more_info):
    """
    This is a basic one - this is a simple solver that will always input the
    100 most frequent clues for which the clue text never returns anything
    but that one answer.
    :param clue_with_more_info: Format is {"clue_text": , "answer_length", "clue_constraint": }
    :return: either None or the answer
    """

    lookup_dict = get_most_common_clue_word_pair()
    return lookup_dict.get(clue_with_more_info["clue_text"], None)


def roman_numeral_possible_solutions(clue_with_more_info):
    """
    Hacky way to just limit the set of solutions for a roman numeral problem
    to only those answers of length answer_length that contain the alphabet
    characters IVCXLMD.

    :param clue_with_more_info: a dictionary like {"clue_text": "", "answer_length": None, "clue_constraint": None}
    :return: possible list of solutions
    """

    answer_length = clue_with_more_info["answer_length"]

    # Get all permutations of letters IVCXLMD of length==answer_length
    possible_answers = [''.join(p) for p in permutations('IVCXLMD', answer_length)]
    return return_constraint_hits(possible_answers, clue_with_more_info)


def directions_possible_solutions(clue_with_more_info):
    """
    Hacky way to just limit the set of solutions for a roman numeral problem
    to only those answers of length answer_length that contain the alphabet
    characters NEWS.

    :param clue_with_more_info: a dictionary like {"clue_text": "", "answer_length": None, "clue_constraint": None}
    :return: possible list of solutions
    """
    answer_length = clue_with_more_info["answer_length"]

    possible_answers = [''.join(p) for p in permutations('NEWS', answer_length)]
    return return_constraint_hits(possible_answers, clue_with_more_info)


def solve_clue(clue_text=None, clue_type=None, clue_with_more_info=None, print_bool=False):
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
    :param clue_with_more_info: optional parameter with richer clue information. Format is
    {"clue_text": , "answer_length", "clue_constraint": }

    :param clue_type: clue type, generally as determined by predict_clues_df_from_default_pipeline() (df) or
                      predict_single_clue_from_default_pipeline
    :param print_bool: if True, print helpful things about results and failures
    :return: a list of answers, normalized into crossword format, or None
    """

    # If clue_text is not None and is a dict, raise an error
    if isinstance(clue_text, dict):
        raise ValueError(
            "clue_text was passed as a dictionary. If passing a dictionary, "
            "use the 'clue_with_more_info' parameter instead.")

    # If clue_with_more_info is not None and isn't a dict, raise an error
    if not isinstance(clue_with_more_info, dict):
        raise ValueError(
            "clue_with_more_info must be a dictionary with keys like 'clue_text', 'answer_length', 'clue_constraint'.")

    # Step 1: Construct full clue_with_more_info if only clue_text is passed
    if clue_with_more_info is None:
        clue_with_more_info = {
            "clue_text": clue_text,
            "answer_length": None,
            "clue_constraint": None
        }
    else:
        # Ensure all keys are present in the dict, fill with fallback values if needed
        # the part that is ALWAYS needed is clue_text!
        clue_with_more_info.setdefault("clue_text", clue_text)
        clue_with_more_info.setdefault("answer_length", None)
        clue_with_more_info.setdefault("clue_constraint", None)

    # Find the appropriate solving methodology
    solving_method = return_methodology_by_clue_type(clue_type)

    # Exit function if solving methodology DNE
    if solving_method is None:
        print_if(statement=f"Clue '{clue_with_more_info['clue_text']}' was attempted to be solved with "
                           f"clue_type = '{clue_type}' but no "
                           f"appropriate solving methodology exists for that clue type.",
                 print_bool=print_bool)
        return None

    # Otherwise, use the method
    else:
        # If there IS a solving methodology, apply it
        statement = f"Clue: '{clue_with_more_info['clue_text']}' is type '{clue_type}'   \n" \
                    f"and will be solved with '{solving_method.__name__}'"
        print_if(statement=statement, print_bool=print_bool)
        answer = solving_method(clue_with_more_info)

        # If answer is None, return None
        if answer is None:
            statement = (f"Clue '{clue_with_more_info['clue_text']}' was attempted to be solved with "
                         f"clue_type={clue_type}, and the methodology {solving_method} was attempted, "
                         f"however there was no return."),
            print_if(statement=statement, print_bool=print_bool)
            return None
        else:
            # If answer is a string, then set candidate_answers = [answer]
            candidate_answers = []
            if isinstance(answer, str):
                candidate_answers.append(answer)

            # If answer is a list, leave as it
            elif isinstance(answer, list):
                candidate_answers = answer

            # Normalize result
            candidate_answers = [process_text_into_clue_answer(x) for x in candidate_answers]
            print_if(statement=f"Candidate answers are {candidate_answers}", print_bool=print_bool)

            return candidate_answers


def stem_recursive_synonyms(recursive_synonyms):
    """
    Given a list of possible synonyms, this modifies that list
    to consider word stems and smaller constituent words.

    :param recursive_synonyms: get_recursive_synonyms output
    :return: list
    """
    new_synonyms = recursive_synonyms  # always include base forms
    for synonym in recursive_synonyms:

        # Also append non-base forms ( xxx tbd could do better at this)
        if synonym.endswith("ing"):
            new_synonyms.append(synonym[:-3])  # stem that -ing

        if synonym.endswith("ed"):
            new_synonyms.append(synonym[:-1])  # stem that -ed to e

        if synonym.endswith("s") and not synonym.endswith("ss"):
            new_synonyms.append(synonym[:-1])  # stem that -s

    return new_synonyms


def solve_clues_dataframe(clues_df, top_n_classes, solve_class_threshold):
    """
    Given a clues dataframe with columns "clue", "word" or "answer (optional column, for checking only)"
    this predicts the top_n_classes and then applies the solving algorithms on
    :param solve_class_threshold: The percentage above which a clue class probability should be solved by that function.
     For predicted classes like [('Acronym/short form answer', 0.16),
                                 ('Example of a class', 0.13),
                                 ('Clever synonyms and examples and examples', 0.12)]
     and solve_class_threshold=0.121, only 'Acronym/short form answer' and 'Example of a class' would get solved

    # created by genAI xxx tbd not working for some clue types due to (I think) how clue_info is being passed

    :param top_n_classes:
    :param clues_df:
    :return:
    """
    predicted_df = predict_clues_df_from_default_pipeline(clues_df=clues_df,
                                                          keep_features=False,
                                                          top_n=top_n_classes)

    solved_answers_list = []

    for _, row in predicted_df.iterrows():
        clue_text = row.get("clue")
        # Try both 'answer (optional column, for checking only)' and 'Word'
        ground_truth_answer = row.get("answer (optional column, for checking only)",
                                      row.get("Word", None))

        answer_length = len(ground_truth_answer) if isinstance(ground_truth_answer, str) else None
        if answer_length is None:
            print(f"[WARNING] No known answer length for clue: '{clue_text}' — some solvers may fail.")

        clue_with_more_info = {
            "clue_text": clue_text,
            "answer_length": answer_length,
            "clue_constraint": None
        }

        candidate_answers = []

        # Loop through predicted classes and solve if probability is above threshold
        for clue_type, prob in row["Top_Predicted_Classes"]:
            if prob >= solve_class_threshold:
                print(f"\n[INFO] Attempting to solve clue: \"{clue_text}\" | type: '{clue_type}' | prob: {prob:.2f}")

                solved = solve_clue(clue_with_more_info=clue_with_more_info,
                                    clue_type=clue_type,
                                    print_bool=False)
                if solved:
                    print(f"[SUCCESS] Found answers: {solved}")

                    candidate_answers.extend(solved)
                else:
                    print(f"[FAILURE] No answers found for type: {clue_type}")

        # Deduplicate and join into string (or keep as list if preferred)
        deduped_answers = list(dict.fromkeys(candidate_answers)) if candidate_answers else None
        solved_answers_list.append(deduped_answers)

    predicted_df["answer_list"] = solved_answers_list
    return predicted_df


''' Workspace 
# Below is the initial attempts to add predicted classes to a single crossword
csv_path = f"{get_project_root()}/data/puzzle_samples/processed_puzzle_samples/crossword_2022_07_24.csv"

crossword = get_crossword_from_csv(csv_path)
all_clues_predicted = predict_clues_df_from_default_pipeline(clues_df=crossword.clue_df,
                                                             keep_features=False,
                                                             top_n=2)

solved_df = solve_clues_dataframe(clues_df=crossword.clue_df, top_n_classes=2, solve_class_threshold=0.8)

# Filter the rows where the 'Top_Predicted_Classes' contains a class with a probability > 0.5
# filtered_clues = all_clues_predicted[
#    all_clues_predicted['Top_Predicted_Classes'].apply(
#        lambda x: any(prob > 0.8 for _, prob in x)  # Check if any probability in the dictionary is greater than 0.5
#    )
# ]

synonyms = [('Gumption', 'NERVE'),
            ('Scatters', 'SOWS'),
            ('Combined', 'MERGED'),
            ('Moor', 'DOCK'),
            ('Bouquet', 'FRAGRANCE'),
            ('Level', 'TIER'),
            ('Suit', 'BEFIT'),
            ('Whiz', 'ACE'),
            ('Edify', 'TEACH'),
            ('Dillydallies', 'DAWDLES'),
            ('Parched', 'DRY'),
            ('Bundles', 'SHEAFS'),
            ('Profess', 'AVER'),
            ('Botch', 'MUFF'),
            ('Loved', 'DEAR'),
            ]

for synonym in synonyms:
    clue, answer = synonym
    solved_answer_list = synonym_wrapper(
        clue_with_more_info={"clue_text": clue, "answer_length": len(answer), "clue_constraint": None})
    print(f"\n\n{clue}\n{answer}\n{solved_answer_list}")


for index, row in all_clues_predicted.iterrows():
    clue_text = row["clue"]
    answer_ = row["answer (optional column, for checking only)"]
    clue_with_more_info_ = {"clue_text": clue_text,
                           "answer_length": len(answer_),
                           "clue_constraint": None}
    clue_types = row["Top_Predicted_Classes"]
    for clue_type, probability in clue_types:
        solved_answer_list = solve_clue(clue_with_more_info=clue_with_more_info_,
                                        clue_type=clue_type,
                                        print_bool=True)



example_synonym_clue = solve_clue(clue_with_more_info={"clue_text": 'happy',
                                                       "answer_length": 6,
                                                       "clue_constraint": None}, clue_type="One word synonym")
example_synonym_clue = solve_clue(clue_with_more_info={"clue_text": 'tiger, in spanish',
                                                       "answer_length": 5,
                                                       "clue_constraint": None}, clue_type="Foreign language")
print(example_synonym_clue)


input_path = os.path.join(get_project_root(), "data/puzzle_samples/processed_puzzle_samples/all_puzzles.csv")
output_folder = os.path.join(get_project_root(), "data/puzzle_samples/solved_clues_batches")
os.makedirs(output_folder, exist_ok=True)
top_n_classes = 3
solve_class_threshold = 0.7
chunk_size = 5

# Load data
df = pd.read_csv(input_path)
total_rows = df.shape[0]

# Process in chunks
for i in range(chunk_size, total_rows + chunk_size, chunk_size):
    print(f"[INFO] Processing rows 0 to {i-1}...")

    # Slice the data
    subset = df.iloc[:i].copy()

    # Solve clues
    solved_df = solve_clues_dataframe(subset, top_n_classes=top_n_classes, solve_class_threshold=solve_class_threshold)

    # Save to Excel
    output_path = os.path.join(output_folder, f"solved_clues_{i}.xlsx")
    solved_df.to_excel(output_path, index=False)
    print(f"[SAVED] {output_path}")
'''