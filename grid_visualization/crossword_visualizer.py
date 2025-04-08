"""
crossword_visualizer.py

This module defines a `CrosswordVisualizer` class that uses Pygame to visually render
a crossword puzzle grid in a New York Times-style layout. The visualizer supports 
dynamic refreshing and is designed to be reusable across different components.
"""

import pygame
import sys
import os
from puzzle_objects.crossword_and_clue import Crossword, get_saved_or_new_crossword


class CrosswordVisualizer:
    def __init__(self, crossword=None):
        """
        Initialize the CrosswordVisualizer.

        Args:
            crossword (Crossword, optional): A Crossword object to visualize. 
                                             If None, loads a saved or new crossword.
        """
        self.crossword = crossword or get_saved_or_new_crossword()[0]
        self._init_visual_attributes()
        self._prepare_crossword_elements()
        self.running = True

    def _init_visual_attributes(self):
        """
        Initialize visual attributes and Pygame settings for the crossword grid display.
        """
        pygame.init()
        self.SCREEN_WIDTH = 1280
        self.SCREEN_HEIGHT = 720

        # Get grid size from the crossword object
        self.GRID_HEIGHT = len(self.crossword.grid)
        self.GRID_WIDTH = len(self.crossword.grid[0])

        # Calculate cell size dynamically based on screen space
        self.CELL_SIZE = min((self.SCREEN_WIDTH - 100) // self.GRID_WIDTH,
                             (self.SCREEN_HEIGHT - 200) // self.GRID_HEIGHT)
        self.MARGIN = 1
        self.WINDOW_WIDTH = self.GRID_WIDTH * (self.CELL_SIZE + self.MARGIN) + self.MARGIN
        self.WINDOW_HEIGHT = self.GRID_HEIGHT * (self.CELL_SIZE + self.MARGIN) + self.MARGIN

        # Create the Pygame screen
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("NYT-Style Crossword Grid")

        # Define colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (200, 200, 200)
        self.LIGHT_BLUE = (173, 216, 230)
        self.LIGHT_GRAY = (230, 230, 230)
        self.GREEN = (150, 240, 150)
        self.RED = (255, 100, 100)
        self.YELLOW = (255, 255, 150)

        # Set fonts for clue numbers and letters
        self.number_font = pygame.font.SysFont('Arial', 12)
        self.letter_font = pygame.font.SysFont('Arial', 24, bold=True)

    def _prepare_crossword_elements(self):
        """
        Process the crossword grid to extract cells, clue numbers, and preset letters.
        """
        # 1 = white cell, 0 = black square
        self.grid = [[1 if cell != "■" else 0 for cell in row] for row in self.crossword.grid]

        # Store clue numbers with their positions
        self.clue_numbers = {
            (row["start_row"], row["start_col"]): int(row["number"])
            for _, row in self.crossword.clue_df.iterrows()
        }

        # Extract pre-filled letters (if any)
        self.letters = {}
        for r in range(len(self.crossword.grid)):
            for c in range(len(self.crossword.grid[0])):
                char = self.crossword.grid[r][c]
                if char != " " and char != "■":
                    self.letters[(r, c)] = char.upper()

    def refresh(self, crossword):
        """
        Refresh the visualizer with a new crossword object and redraw the grid.

        Args:
            crossword (Crossword): New crossword object to display.
        """
        self.crossword = crossword
        self._prepare_crossword_elements()
        self.draw_grid()
        pygame.display.flip()

    def draw_grid(self):
        """
        Draw the entire crossword grid with clue numbers and letters.
        """
        self.screen.fill(self.GRAY)
        for row in range(self.GRID_HEIGHT):
            for col in range(self.GRID_WIDTH):
                x = col * (self.CELL_SIZE + self.MARGIN) + self.MARGIN
                y = row * (self.CELL_SIZE + self.MARGIN) + self.MARGIN

                # Determine cell color
                if self.grid[row][col] == 0:
                    color = self.BLACK
                elif (row, col) in self.clue_numbers:
                    color = self.LIGHT_GRAY
                else:
                    color = self.WHITE

                # Draw cell rectangle
                pygame.draw.rect(self.screen, color, [x, y, self.CELL_SIZE, self.CELL_SIZE])

                # Draw clue number (top-left of the cell)
                if (row, col) in self.clue_numbers:
                    number_text = self.number_font.render(str(self.clue_numbers[(row, col)]), True, self.BLACK)
                    self.screen.blit(number_text, (x + 2, y + 2))

                # Draw letter (center of the cell)
                if (row, col) in self.letters:
                    letter_text = self.letter_font.render(self.letters[(row, col)], True, self.BLACK)
                    text_rect = letter_text.get_rect(center=(x + self.CELL_SIZE // 2, y + self.CELL_SIZE // 2 + 3))
                    self.screen.blit(letter_text, text_rect)

    def run(self):
        """
        Start the main Pygame loop to display and interact with the crossword.
        Press 'Q' to quit the window.
        """
        print("[INFO] Crossword Visualizer Running. Press Q to quit.")
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        self.running = False
            self.draw_grid()
            pygame.display.flip()

        pygame.quit()
        sys.exit()
