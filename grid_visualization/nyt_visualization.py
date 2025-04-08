"""
nyt_visualization.py

This script is the entry point to launch a graphical crossword visualizer.
It loads a crossword (either a saved one or a new one), then starts the visual interface
using the CrosswordVisualizer class.

"""

import sys
import os

# Add the parent directory to the module search path so that we can import project files
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the crossword retrieval function and visualizer class
from puzzle_objects.crossword_and_clue import get_saved_or_new_crossword
from grid_visualization.crossword_visualizer import CrosswordVisualizer

if __name__ == "__main__":
    # Load a crossword puzzle and its filename (could be a previously saved puzzle)
    crossword, filename = get_saved_or_new_crossword()

    # Create a visualizer instance using the loaded crossword puzzle
    visualizer = CrosswordVisualizer(crossword)

    # Start the visualizer, which opens a Pygame window showing the grid
    visualizer.run()
