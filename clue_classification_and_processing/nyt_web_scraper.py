import time
import calendar
import datetime
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from clue_classification_and_processing.helpers import get_project_root

"""
This file contains functions which support 
"""

def get_day_of_week(date_obj):
    return calendar.day_name[date_obj.weekday()].lower()

def format_filename(date_obj):
    day_name = get_day_of_week(date_obj)
    return f"{day_name}_{date_obj.strftime('%m%d%Y')}.html"

def download_and_reveal_puzzle(date_str):
    """
    Automates downloading and revealing a NYT crossword puzzle via Selenium.

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

# example usage:
# download_and_reveal_puzzle("2025-02-04")