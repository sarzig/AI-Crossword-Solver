# New York Times A.I. Crossword Solver - CS5100 Foundations of A.I. Project

Eroniction~ Can you please add a gif of the visualization right here? !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

## Team Members
- Eroniction
- Sarah
- Sheryl
- Swathi

## Project Overview and Motivation
Our team of 4 members (Eroniction, Sarah, Sheryl, and Swathi) are all fans of games, puzzles, and natural language processing. For our semester-long CS5100 Foundation of AI project, we are creating an AI agent that tackles the most pressing challenge one can undertake in their pajamas with a cup of coffee in hand: The New York Times crossword.

## Getting Started: how to run this project 
1. Install requirements from requirements.txt
2. Download BERT model from xxx Sheryl xxx !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
3. Ensure project root directory is named "ai_crossword_solver"
4. Run demo.py to be taken on an interactive walk through 

## GitHub Structure
* ai_crossword_solver (root / project directory folder) 
  * demo.py - the step-by-step examples of the various key parts of our project
  * requirements.txt - the requirements file you'll need to run our code
  * clue_classification_and_processing 
    * Where k-means and machine learning work occurred to *classify* clues by clue type
  * clue_solving
    * Clue solving algorithms (Wikipedia search, foreign language, synonym)
  * data
    * NYT dataset (nytcrosswords.csv)
    * puzzle samples (raw HTML from web scraping and processed CSVs)
  * grid_visualization
    * PyGame visualization framework
  * puzzle_objects
    * The "Crossword" data structure, subsetting algorithms, and clue placing algorithms
  * tests
    * PyTest tests
  * testing_results
    * CSP, Bert, and other test results
