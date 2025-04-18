"""
This is the demo file that we recommend our TA use to see the core functions
of our project.

It does the following:
 * gets crossword object from a file
 * creates a Crossword object
        - and shows quick "detailed print"
 * does prediction of clue types using clue_classification_ml_pipeline
        - and prints out the summary of that dataFrame to show which clues were solved
 * Creates a CrosswordVisualizer object to help view the crossword
 * Calls constraint solver to apply the given board constraints

"""
import os
import time
import pandas as pd
from tabulate import tabulate
from clue_classification_and_processing.helpers import get_project_root, header_print
from puzzle_objects.crossword_and_clue import get_crossword_from_csv

######################################################################################################
# Get crossword object from file
######################################################################################################
header_print("Getting crossword object from file")
time.sleep(2)
crossword_csv = "crossword_2022_07_24.csv"
crossword_full_path = os.path.join(get_project_root(),
                                   "data",
                                   "puzzle_samples",
                                   "processed_puzzle_samples",
                                   crossword_csv)
dataframe_from_csv = pd.read_csv(crossword_full_path)
crossword = get_crossword_from_csv(crossword_full_path)

print("Dataframe contents (this represents what was web scraped from the NYT Crossword website):")
time.sleep(2)
print(tabulate(dataframe_from_csv,
               headers=dataframe_from_csv.columns,
               tablefmt="rounded_outline",
               showindex=False))
time.sleep(2)


######################################################################################################
# Create a Crossword object and show "detailed print"
######################################################################################################
header_print('# Create a Crossword object and show the "detailed print" of the empty grid')
time.sleep(2)
crossword = get_crossword_from_csv(crossword_full_path)
crossword.detailed_print()
time.sleep(2)

######################################################################################################
# Show the visualization of the crossword - Eroniction
######################################################################################################
# Eroniction - can you please place some visualization piece here to demo?

######################################################################################################
# Add multi-word and OOV (out of vocabulary) words
######################################################################################################
header_print('# Add multi-word and OOV words')
time.sleep(2)
print("Adding in multi-word answers, such as HEROWORSHIPPER, EVERGREENTERRACE, and ALLROADSLEADTOROME")
print("was essential to beginning the CSP approach.")
time.sleep(2)
crossword.place_helper_answers(fill_oov=True)
crossword.detailed_print()


######################################################################################################
# Prediction of clue types and initial solving
######################################################################################################
header_print('# Prediction of clue types and initial solving')
print("Note that no solving algorithms yield a 100% correct, guaranteed answer.\n"
      "Inherent ambiguity, mis-directions, and the large possible set of answers\n"
      "for any given clue make a LIST format much more appropriate."
      ""
      "\n\nClue solving can take up to 3 minutes ...")
time.sleep(1)
crossword.solve_self_clues_dataframe(print_bool=True)
subset_to_show = crossword.clue_df.copy(deep=True)
print(tabulate(dataframe_from_csv,
               headers=dataframe_from_csv.columns,
               tablefmt="rounded_outline",
               showindex=False))

filtered_df = crossword.clue_df[
    crossword.clue_df['answer_list'].notna() & (crossword.clue_df['answer_list'].str.len() > 0)
]

# Rename the columns as specified
filtered_df_renamed = filtered_df.rename(columns={
    'number_direction': '#',
    'clue': 'Clue',
    'answer (optional column, for checking only)': 'Actual Answer',
    'answer_list': 'Potential Answer List'
})

print("These are the potential answer lists of the clues which DID have solving algorithms applied.\n")
print(tabulate(filtered_df_renamed[['#', 'Clue', 'Actual Answer', "Top_Predicted_Classes", 'Potential Answer List']],
               headers='keys', tablefmt='rounded_outline', showindex=False))


######################################################################################################
# CSP and Bert - Sheryl
######################################################################################################


