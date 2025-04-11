import time
import calendar
import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from clue_classification_and_processing.helpers import get_project_root

"""
This file contains functions which support automated retrieval of NYT crossword puzzles. 
Ultimately most of these were NOT used because of the New York Times detection of 
automatic web-scrapers. I instead downloaded the full html on 200 crosswords, saved those
as .htmls and then processed them. (See 

Functions:
----------
- get_day_of_week(date_obj): 
    Returns the lowercase day of the week for a given datetime.date or datetime.datetime object.

- format_filename(date_obj): 
    Formats a filename using the day of the week and date, e.g., "tuesday_02042025.html".

- download_and_reveal_puzzle(date_str): 
    Opens the NYT Crossword puzzle for a given date, prompts the user to log in,
    reveals the full puzzle using Selenium, and saves the HTML to the local project folder.
"""


def get_day_of_week(date_obj):
    """
    Extract day of week from date object. This is used for checking which
    day of the week a puzzle is from. Mondays are easiest and smallest, and
    Sundays are hardest and largest.
    :param date_obj: date
    :return: day of the week, like "monday"
    """
    return calendar.day_name[date_obj.weekday()].lower()


def format_filename(date_obj):
    """
    Given a day, create a formatted html filename.
    :param date_obj: date object
    :return: the filename (.html)
    """
    day_name = get_day_of_week(date_obj)
    return f"{day_name}_{date_obj.strftime('%m%d%Y')}.html"


def download_and_reveal_puzzle(date_str):
    """
    Automates downloading and revealing a NYT crossword puzzle via Selenium.

    example usage:
    download_and_reveal_puzzle("2025-02-04")

    gen ai

    Also note, this didn't work out - NYT is savvy to web scrapers and
    threatens to ban my account. Thus, I manually downloaded things.

    :param date_str: Date in 'YYYY-MM-DD' format, e.g., '2025-02-04'
    """
    date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')
    url = f"https://www.nytimes.com/crosswords/game/daily/{date_obj.strftime('%Y/%m/%d')}"

    # Start browser
    driver = webdriver.Chrome()  # Or use Firefox, Edge, etc.
    driver.get(url)

    # Prompt user to log in
    print("🔐 Please log in to the NYT Crossword site in the opened browser.")
    input("Type 'y' when you are logged in and the puzzle is fully visible: ")

    try:
        # Step 1: Click Reveal button
        reveal_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='reveal']"))
        )
        reveal_btn.click()
        time.sleep(1)

        # Step 2: Click "Puzzle" in reveal menu
        puzzle_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[text()='Puzzle']"))
        )
        puzzle_btn.click()
        time.sleep(1)

        # Step 3: Confirm Reveal
        confirm_btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Reveal']"))
        )
        confirm_btn.click()

        # Wait for reveal animation/data to load
        time.sleep(2)

        # Step 4: Save page HTML
        html = driver.page_source
        output_folder = f'{get_project_root()}/data/puzzle_samples/raw_html'
        output_folder.mkdir(parents=True, exist_ok=True)
        filename = format_filename(date_obj)
        output_path = output_folder / filename

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"✅ Puzzle saved to: {output_path}")

    except Exception as e:
        print(f"❌ Something went wrong: {e}")

    finally:
        driver.quit()