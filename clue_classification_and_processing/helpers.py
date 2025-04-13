"""
Author: Sarah

This is the generic helpers file. 

Generally functions will be put here if they work well, do not have a better "home" 
elsewhere, and are polished.

Summary of functions-----------------------------------------------
General helpers:
* print_if(statement, print_bool)
* conditional_raise(error, raise_bool)

Get project specific resources
* def get_project_root()
* get_clues_dataframe(clues_path=None)

Text processing
* preprocess_lower_remove_punct_strip_whitespace(input_text)
* process_text_into_clue_answer(input_text)
-------------------------------------------------------------------
"""

import os
import re
import string
import pandas as pd
import hashlib

from clue_classification_and_processing.clue_classification_machine_learning import \
    predict_clues_df_from_default_pipeline


def print_if(statement, print_bool):
    """
    Helper to simplify conditional printing.

    :param statement: statement to print
    :param print_bool: boolean to print or not
    """
    if print_bool:
        print(statement)


def conditional_raise(error, raise_bool):
    """
    Helper to simplify the conditional raising of errors.

    :param error: error to raise
    :param raise_bool: boolean to raise
    """

    if raise_bool:
        raise error


def cool_error(error):
    """
    Print a very large error to hopefully convince the user to notice urgent
    action should be taken.
    :return: nothing
    """

    error_text = \
        """
    +-------------------------------------------------+
    |     .d88b.  888d888 888d888  .d88b.  888d888    |
    |    d8P  Y8b 888P"   888P"   d88""88b 888P"      |
    |    88888888 888     888     888  888 888        |
    |    Y8b.     888     888     Y88..88P 888        |
    |     "Y8888  888     888      "Y88P"  888        |
    +-------------------------------------------------+    
     """

    print(error_text)
    raise error


def get_clues_by_class(clue_class="all", classification_type="manual_only", prediction_threshold=None):
    """
    This queries two datasets:
      * nyt_crosswords.csv
      * Sarah's manually classified clues

    If classification_type is manual_only, then clues of the given clue class (or ALL classes with manual
    classes) will be returned in a df.

    If classification type is predicted_only, then the full kaggle dataset will be queried, with the
    predictions applied by my ML model. Beware, these are frequently incorrect, especially in pretty critical
    categories like "straight definition".

    :param: clue_class = if all, gives all clue types
    :param: classification_type= "manual_only", "predicted_only", "all"
    :return: df with columns ["clue", "Word", "Class"]. Approximately 5k manually classed rows and 700k ML classed rows
    """

    loc = ""
    text = ""

    # If only looking for manually classed clues, look in
    # the manually classified clues.xlsx
    if classification_type == "manual_only":
        text = "manual"
        loc = os.path.join(get_project_root(),
                           "data",
                           "clue_classification_ml_pipeline",
                           "all_manually_classified_clues.xlsx")

        # read the dataframe from the location
        df = pd.read_excel(loc)
        class_series = (df["Class"]).dropna()
        class_series = class_series[class_series.apply(lambda x: isinstance(x, str))]
        classes = sorted(set(class_series))
        print(f"\nPulling {text} classified clues from\n{loc}.")

        # If class is not all, then subset to that class
        if clue_class == "all":
            print("Returning clues of all classes.\n")
        elif clue_class in classes:
            df = df[df["Class"] == clue_class]
        else:
            print("Unrecognized class. Please select a class from list:")
            classes = get_class_options()
            for each in classes:
                print(f"  * {each}")
            return None

        # Get only columns of interest
        columns_of_interest = ["clue", "Word", "Class", "Confidence"]
        available_columns = [col for col in columns_of_interest if col in df.columns]
        df = df[available_columns].copy()

        # Make sure all columns in df are strings
        for col in df.columns:
            if col != "Confidence":
                df[col] = df[col].astype(str)

        return df

    # If looking for only predicted, then just use the full_clue set and assign predictions
    # Only get clues that have prediction threshold over 0.8
    if classification_type == "predicted_only":
        if prediction_threshold is None:
            prediction_threshold = 0

        text = "ML"
        loc = os.path.join(get_project_root(),
                           "data",
                           "clue_classification_ml_pipeline",
                           "all_clues_predicted.csv")

        df = pd.read_csv(loc)

        # Ensure Top_Predicted_Classes is parsed if it's a string
        if isinstance(df["Top_Predicted_Classes"].iloc[0], str):
            import ast
            df["Top_Predicted_Classes"] = df["Top_Predicted_Classes"].apply(ast.literal_eval)

        # If specific clue class, filter and extract probability
        if clue_class != "all":
            def extract_prob(predictions):
                for cls, prob in predictions:
                    if cls == clue_class:
                        return prob
                return 0  # if not found

            df["class_probability"] = df["Top_Predicted_Classes"].apply(extract_prob)
            df = df[df["class_probability"] >= prediction_threshold]
            df = df.sort_values("class_probability", ascending=False)
            print(f"Returning ML-classified clues for class '{clue_class}' with threshold ≥ {prediction_threshold}\n")
        else:
            print("Returning ML-classified clues for all classes.\n")

        # Keep only relevant columns
        columns_of_interest = ["clue", "Word", "Top_Predicted_Classes"]
        if clue_class != "all":
            columns_of_interest.append("class_probability")

        available_columns = [col for col in columns_of_interest if col in df.columns]
        df = df[available_columns].copy()

        for col in ["clue", "Word"]:
            if col in df.columns:
                df[col] = df[col].astype(str)

        return df


def get_class_options():
    """
    Simply looks into Sarah's manually classed clues and returns a list of all classes.

    :return: list of classes
    """
    manual_clues = get_clues_by_class(clue_class="all", classification_type="manual_only")
    unique_clues = list(set((manual_clues["Class"].to_list())))
    unique_clues.sort()
    return unique_clues


def get_vocab():
    """
    Fetch vocab from the combined_vocab.txt file.
    :return: set of the vocab
    """
    try:
        location = os.path.join(get_project_root(), "data", "combined_vocab.txt")
        with open(location, "r", encoding="utf-8") as f:
            print(f"Fetching combined vocab (nltk words, NYT data) from {location}")
            return set(line.strip() for line in f if line.strip())

    except Exception:
        location = os.path.join(get_project_root(), "combined_vocab.txt")
        with open(location, "r", encoding="utf-8") as f:
            print(f"Fetching combined vocab (nltk words, NYT data) from {location}")
            return set(line.strip() for line in f if line.strip())


def stable_hash(obj):
    """
    Stable hash will return the same random value every single time.

    :param obj: object
    :return: hashed integer
    """
    return int(hashlib.md5(str(obj).encode()).hexdigest(), 16)


def get_project_root():
    """
    Uses OS lib to search for cwd, and then walks back to project root.

    :return: os.path object
    """
    # get cwd and split into constituent parts
    cwd = os.getcwd()
    path_parts = cwd.split(os.sep)

    # Look for project name in the path
    project_root = ""
    if "ai_crossword_solver" in path_parts:
        index = path_parts.index("ai_crossword_solver")
        project_root = os.sep.join(path_parts[:index + 1])

    # If ai_crossword_solver isn't anywhere in the path, then flag
    else:
        error_text = "To our TA: Please note the parent project directory expects to be named 'ai_crossword_solver'"
        cool_error(FileNotFoundError(error_text))

    return project_root


def get_processed_puzzle_sample_root():
    """
    Quick helper to get the path to processed_puzzle_samples.
    :return: an os.path object
    """
    return os.path.join(get_project_root(),
                        "data",
                        "puzzle_samples",
                        "processed_puzzle_samples")


def get_clues_dataframe(clues_path=None, delete_dupes=False):
    """
    Uses OS lib to search for cwd, and then walks back to project root.

    Alternately, if you give it a path it just pulls from that.

    :return: the main Kaggle dataframe with all clues
    """

    if clues_path is None:
        # get cwd and split into constituent parts
        cwd = os.getcwd()
        path_parts = cwd.split(os.sep)

        # Look for project name in the path
        root = ""
        if "ai_crossword_solver" in path_parts:
            index = path_parts.index("ai_crossword_solver")
            root = os.sep.join(path_parts[:index + 1])

        # Load dataset
        clues_path = os.path.join(root, r"data", "nytcrosswords.csv")

    # Return the dataframe from that csv
    clues_df = pd.read_csv(clues_path, encoding='latin1')
    if delete_dupes:
        clues_df = clues_df.drop_duplicates(["Word", "clue"])
    return clues_df


def get_100_most_common_clues_and_answers():
    """
    Every savvy cross-worder knows that "Actress Thurman" resolves a pesky puzzle triplet,
    that Jai Alai is a beautifully voweled Basque sport, and that Tae Kwon Do is a
    respected martial art.

    This function returns the 100 most common clues and answers, which we assume
    a person solving the crossword would know by good-old-fashioned rote memorization.

    # Ai assisted

    :return: dataframe with columns Clue, Word, count, and is_unique_clue
    """
    clues_df = get_clues_dataframe()

    # Top 200 most common Clue–Word pairs
    common_pairs = (
        clues_df.groupby(["clue", "Word"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(200)
    )

    # All unique (Clue, Word) pairs in top clues
    filtered_clues_df = (
        clues_df[clues_df["clue"].isin(common_pairs["clue"])]
        .drop_duplicates(subset=["clue", "Word"])
    )

    # Clues that only appear once among top pairs
    unique_clues_set = (
        filtered_clues_df["clue"]
        .value_counts()
        .loc[lambda x: x == 1]
        .index
    )

    # Mark each Clue–Word pair in top 200 as unique or not
    common_pairs["is_unique_clue"] = common_pairs["clue"].isin(unique_clues_set)
    common_pairs = common_pairs[common_pairs["is_unique_clue"] is True]  # only subset the clues we care about

    return common_pairs


def preprocess_lower_remove_punct_strip_whitespace(input_text):
    """
    Lowers case of input, replaces all white space and punctuation with " ".

    :param input_text: text to modify
    :return: new text
    """
    new_text = input_text.lower()

    # Remove punctuation that is NOT within a word (preserve in-word punctuation like "john's" -> "johns" and
    # "honky-tonk" -> "honkytonk")
    new_text = re.sub(r'\b(\w+)[\'-](\w+)\b', r'\1\2', new_text)  # Merge words with apostrophe or hyphen
    new_text = re.sub(fr"[{re.escape(string.punctuation)}]", " ", new_text)  # Remove other punctuation

    # Normalize whitespace
    new_text = re.sub(r'\s+', ' ', new_text).strip()

    return new_text


def process_text_into_clue_answer(input_text):
    """
    Removes all white space, converts characters into English equivalent.

    :param input_text: input_text to process into a clue answer
    :return: processed text
    """

    # Replace all possible whitespace in clue with nothing
    whitespace_regex = r"[\s\u00A0\u1680\u180E\u2000-\u200A\u2028\u2029\u202F\u205F\u3000]"

    # These are all special characters which could theoretically
    # come up in the source pages of an answer. They need to be Anglicized into their closest
    # english approximation (a, e, i, o, u, y, n).
    replace_special_letters = {
        "a": ["á", "à", "â", "ä", "ã", "å", "ā", "ă", "ą", "ȧ", "ǎ"],
        "e": ["é", "è", "ê", "ë", "ē", "ĕ", "ė", "ę", "ě"],
        "i": ["í", "ì", "î", "ï", "ī", "ĭ", "į", "ı", "ȉ", "ȋ"],
        "o": ["ó", "ò", "ô", "ö", "õ", "ō", "ŏ", "ő", "ȯ", "ȱ", "ø"],
        "u": ["ú", "ù", "û", "ü", "ũ", "ū", "ŭ", "ů", "ű", "ų", "ȕ", "ȗ"],
        "y": ["ý", "ÿ", "ŷ", "ȳ", "ɏ"],
        "n": ["ñ", "ń", "ņ", "ň", "ŉ", "ŋ"]
    }

    # remove punctuation
    new_text = input_text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

    # Remove special letters by iterating across the dict.
    for base_letter, variants in replace_special_letters.items():
        for variant in variants:
            new_text = new_text.replace(variant, base_letter)

    # remove whitespace and lowercase it
    new_text = re.sub(whitespace_regex, "", new_text).upper()

    return new_text


def get_most_common_clue_word_pair():
    """
    In this dictionary the clues are the dictionary keys and the
    values are the answers.

    :return: dictionary
    """

    most_frequent_clue_answer_pairs = {
        "Jai ___": "ALAI",
        "___ Lingus": "AER",
        '"Dies ___"': "IRAE",
        "___-Magnon": "CRO",
        "Actress Thurman": "UMA",
        "___ Lanka": "SRI",
        "Neither's partner": "NOR",
        "___ avis": "RARA",
        "Coach Parseghian": "ARA",
        "Inter ___": "ALIA",
        "___-mo": "SLO",
        "From ___ Z": "ATO",
        "___ polloi": "HOI",
        "Rap's Dr. ___": "DRE",
        "Actor Morales": "ESAI",
        "___ Na Na": "SHA",
        "Santa ___ winds": "ANA",
        "___'acte": 'ENTR',
        "Director Kazan": "ELIA",
        "___-majeste": "LESE",
        "Singer Sumac": "YMA",
        "Sugar suffix": "OSE",
        "Directional suffix": "ERN",
        "___ vera": "ALOE",
        "Squeeze (out)": "EKE",
        "___ nous": "ENTRE",
        "Actress Gardner": "AVA",
        "Sault ___ Marie": "STE",
        "___ facto": "IPSO",
        "___ pro nobis": "ORA",
        "Killer whale": "ORCA",
        "Nobelist Wiesel": "ELIE",
        "___ Paulo, Brazil": "SAO",
        "Meadow": "LEA",
        '"Exodus" hero': "ARI",
        "Prefix with center": "EPI",
        "___ kwon do": "TAE",
        "Singer DiFranco": "ANI",
        "___ Park, Colo.": "ESTES",
        "___ Aviv": "TEL",
        "Son of Seth": "ENOS",
        '"Norma ___"': "RAE",
        "Mao ___-tung": "TSE",
        "Suffix with buck": "AROO",
        "___ culpa": "MEA",
        "___ Bator": "ULAN",
        "___ Jima": "IWO",
        "Jacob's twin": "ESAU",
        "Norway's capital": "OSLO",
        "___ buco": "OSSO",
        "___ Stanley Gardner": "ERLE",
        "___ Tin Tin": "RIN",
        "Actress Sommer": "ELKE",
        "___ Mahal": "TAJ",
        '"Mazel ___!"': "TOV",
        "Cosmetician Lauder": "ESTEE",
        "Tel ___": "AVIV",
        '"Como ___ usted?"': "ESTA",
        '"___ Mio"': "OSOLE",
        "Musician Brian": "ENO",
        "Enzyme suffix": "ASE",
        "Afore": "ERE",
        "Once, once": "ERST",
        "Pie ___ mode": "ALA",
        "___ v. Wade": "ROE",
        "Actress Ward": "SELA",
        "Raison d'___": "ETRE",
        "Russia's ___ Mountains": "URAL",
        "Writer ___ Stanley Gardner": "ERLE",
        '"Auld Lang ___"': "SYNE",
        "Mata ___": "HARI",
        "Teachers' org.": "NEA",
        "Author LeShan": "EDA",
        "Photographer Adams": "ANSEL",
        "Suit to ___": "ATEE",
        "___-cone": "SNO",
        "Designer Cassini": "OLEG",
        "Villa d'___": "ESTE",
        "Tempe sch.": "ASU",
        "Actress Skye": "IONE",
        "Up to the task": "ABLE",
        "Gymnast Korbut": "OLGA",
        "Singer McEntire": "REBA",
        "Composer Stravinsky": "IGOR",
        "Miracle-___": "GRO",
        "At any time": "EVER",
        "Sitarist Shankar": "RAVI",
        "Taj Mahal site": "AGRA",
        "For fear that": "LEST",
        "___ Moines": "DES",
        "Anthem contraction": "OER",
        "Buenos ___": "AIRES",
        "Author Calvino": "ITALO",
        "Vicinity": "AREA",
        "Dernier ___": "CRI",
        "Rat-___": "ATAT",
        "QB Manning": "ELI",
        "Up to, informally": "TIL",
        "___-Cat": "SNO",
        "Singer Guthrie": "ARLO"}

    """
    # Backups - the next set
    ""Othello" villain": "IAGO",
    "Mrs. Chaplin": "OONA",
    "___ Canals": "SOO",
    "___ Paulo": "SAO",
    "Humorist Bombeck": "ERMA",
    "Fish eggs": "ROE",
    "Pitcher Hershiser": "OREL",
    "Stead": "LIEU",
    "___ B'rith": "BNAI",
    ""Mamma ___!"": "MIA",
    "Actress Hatcher": "TERI",
    "Messenger ___": "RNA",
    ""The Time Machine" people": "ELOI",
    "___ Gay": "ENOLA",
    "Actress Hagen": "UTA",
    "Soak (up)": "SOP",
    "___-jongg": "MAH",
    "Florence's river": "ARNO",
    "Inventor Whitney": "ELI",
    ""___ Lisa"": "MONA",
    "Baton Rouge sch.": "LSU",
    "Lawyers' org.": "ABA",
    "Tennis's Nastase": "ILIE",
    "Unit of force": "DYNE",
    ""Picnic" playwright": "INGE",
    "Day-___": "GLO",
    ""___ Miserables"": "LES",
    "Linguist Chomsky": "NOAM",
    ""Rule, Britannia" composer": "ARNE",
    "Ginger ___": "ALE",
    "Privy to": "INON",
    "___-do-well": "NEER",
    "Tabula ___": "RASA",
    "Actress Garr": "TERI",
    "Author Rand": "AYN",
    "___-Ball": "SKEE",
    "To the ___ degree": "NTH",
    "___ vu": "DEJA",
    "Lotion ingredient": "ALOE",
    "Fannie ___": "MAE",
    ""___ luck!"": "LOTSA",
    "Asian holiday": "TET",
    "___ Domini": "ANNO",
    "Actress Vardalos": "NIA",
    "___ Gay (W.W. II plane)": "ENOLA",
    "Composer Satie": "ERIK",
    "General ___ chicken": "TSOS",
    "Comic Philips": "EMO",
    ""Zip-___-Doo-Dah"": "ADEE",
    "___ chi": "TAI",
    ""Bali ___"": "HAI",
    "Stephen of "The Crying Game"": "REA",
    "Artist Magritte": "RENE",
    "___ Beta Kappa": "PHI",
    "___ colada": "PINA",
    "___ Reader": "UTNE",
    "Date": "SEE",
    "Director Kurosawa": "AKIRA",
    "Coeur d'___, Idaho": "ALENE",
    "Don Juan's mother": "INEZ",
    "Genesis garden": "EDEN",
    "Verdi's "___ tu"": "ERI",
    "___ Miss": "OLE",
    "1998 Sarah McLachlan hit": "ADIA",
    "Golfer Aoki": "ISAO",
    "Bay window": "ORIEL",
    "Mentalist Geller": "URI",
    "Needle case": "ETUI",
    "Church recess": "APSE",
    "Mountain nymph": "OREAD",
    "___-Rooter": "ROTO",
    "Prefix with dynamic": "AERO",
    "Food thickener": "AGAR",
    "Latvia's capital": "RIGA",
    "Poet ___ St. Vincent Millay": "EDNA",
    "___ Speedwagon": "REO",
    "Morales of "La Bamba"": "ESAI",
    "Spanish gold": "ORO",
    ""Star Wars" princess": "LEIA",
    "Skater Midori": "ITO",
    "Together, in music": "ADUE",
    "Actress Peeples": "NIA",
    "Wild guess": "STAB",
    "Part of A.D.": "ANNO",
    "Some E.R. cases": "ODS",
    "Margarine": "OLEO",
    "Elvis's middle name": "ARON",
    "Violinist Leopold": "AUER",
    "Beethoven's Third": "EROICA",
    "Take to court": "SUE",
    "Supply-and-demand subj.": "ECON",
    "Actor Beatty": "NED",
    "Gen. Robt. ___": "ELEE",
    "Gen. Bradley": "OMAR",
    "Actor Stephen": "REA",
    "Exist": "ARE",
    "Book after Joel": "AMOS",
    "Cleveland's lake": "ERIE",
    "Author Ferber": "EDNA",
    "Before, in poetry": "ERE",
    "___-Ball (arcade game)": "SKEE",
    "Baseball family name": "ALOU",
    "Feudal lord": "LIEGE",
    "Golfer Ballesteros": "SEVE",
    "Actress Lena": "OLIN",
    "Tennis call": "LET",
    "Actress Turner": "LANA",
    "Israel's Abba": "EBAN",
    "Barely make, with "out"": "EKE",
    "Seep": "OOZE",
    ""The Thin Man" dog": "ASTA",
    "Touched down": "ALIT",
    "___ of Man": "ISLE",
    "Nabisco cookie": "OREO",
    "Justice Kagan": "ELENA",
    "Border on": "ABUT",
    "Algerian port": "ORAN",
    "Composer Khachaturian": "ARAM",
    "Stimpy's TV pal": "REN",
    "Western treaty grp.": "OAS",
    "___ Lama": "DALAI",
    "Singer James": "ETTA",
    "Spanish bear": "OSO",
    "Suffix with pay": "OLA",
    "Mrs. Gorbachev": "RAISA",
    "___ about (approximately)": "ONOR",
    "San ___, Italy": "REMO",
    ""... ___ saw Elba"": "EREI",
    "Chicken ___ king": "ALA",
    ""The Time Machine" race": "ELOI",
    "Opposite of paleo-": "NEO",
    "Sicilian spouter": "ETNA",
    "Before, in verse": "ERE",
    "Perlman of "Cheers"": "RHEA",
    "Philosopher Descartes": "RENE",
    "Work units": "ERGS",
    "Memo abbr.": "ATTN",
    "Grad": "ALUM",
    "Composer Bartok": "BELA",
    "Speechify": "ORATE",
    "Actor Milo": "OSHEA",
    "___-di-dah": "LAH",
    "Foxy": "SLY",
    "Simplicity": "EASE",
    "___ tide": "NEAP",
    ""How was ___ know?"": "ITO",
    "Bird: Prefix": "AVI",
    "Actress Verdugo": "ELENA",
    ""___ tu" (Verdi aria)": "ERI",
    "Tater": "SPUD",
    ""I cannot tell ___"": "ALIE",
    "San Francisco's ___ Hill": "NOB",
    "China's Zhou ___": "ENLAI",
    ""___, Brute?"": "ETTU",
    ""___ kleine Nachtmusik"": "EINE",
    "Actor Davis": "OSSIE",
    "___ king": "ALA",
    "Minneapolis suburb": "EDINA",
    "Make sense": "ADDUP",
    ""... ___ quit!"": "ORI",
    ""___ had it!"": "IVE",
    "Pearl Buck heroine": "OLAN",
    "Capek play": "RUR",
    ""___ la Douce"": "IRMA",
    "Inventor Howe": "ELIAS",
    "Vietnam's capital": "HANOI",
    "Designer Geoffrey": "BEENE",
    "French seasoning": "SEL",
    ""Peter Pan" dog": "NANA",
    "Art ___": "DECO",
    "Cambodia's Lon ___": "NOL",
    "Diarist Nin": "ANAIS",
    "Falco of "The Sopranos"": "EDIE",
    "Writer Wiesel": "ELIE",
    "Actress ___ Dawn Chong": "RAE",
    "___ es Salaam": "DAR",
    "___ voce": "SOTTO",
    "Pince-___": "NEZ",
    "Hydroxyl compound": "ENOL",
    "Big rig": "SEMI",
    "Langston Hughes poem": "ITOO",
    ""___ bin ein Berliner"": "ICH",
    "___ many words": "INSO",
    "La ___ Tar Pits": "BREA",
    "Before, poetically": "ERE",
    ""Beetle Bailey" dog": "OTTO",
    ""___ Tu" (1974 hit)": "ERES",
    "Wife of Jacob": "LEAH",
    "China's Chou En-___": "LAI",
    "Itinerary word": "VIA",
    "Hotelier Helmsley": "LEONA",
    "Wonderment": "AWE",
    ""Ben-___"": "HUR",
    "Beethoven dedicatee": "ELISE",
    "Poet Teasdale": "SARA",
    "Actress Rowlands": "GENA",
    "Prefix with friendly": "ECO",
    "___-European": "INDO",
    "Doll's cry": "MAMA",
    "Fly high": "SOAR",
    "Actor Guinness": "ALEC",
    "___ salts": "EPSOM",
    "Applications": "USES",
    "Zhou ___": "ENLAI",
    "Withered": "SERE",
    "End of ___": "ANERA",
    ""Comin' ___ the Rye"": "THRO",
    "Church council": "SYNOD",
    "Maiden name preceder": "NEE",
    "Arctic explorer John": "RAE",
    "Chop ___": "SUEY",
    "Debussy's "La ___"": "MER",
    "Mother-of-pearl": "NACRE",
    "Diva's delivery": "ARIA",
    "Letters before an alias": "AKA",
    "Price abbr.": "CTS",
    "Alias": "AKA",
    "Susan of "L.A. Law"": "DEY",
    "Scruff": "NAPE",
    "Les ___-Unis": "ETATS",
    ""Vive ___!"": "LEROI",
    "Sky-blue": "AZURE",
    "Hawaii's state bird": "NENE",
    "Actress Massey": "ILONA",
    ""Sprechen ___ Deutsch?"": "SIE",
    "Gallic girlfriend": "AMIE",
    "Gaelic": "ERSE",
    "Golfer Ernie": "ELS",
    "___ dixit": "IPSE",
    ""Me neither"": "NORI",
    "Suffix with switch": "EROO",
    "Rock's Motley ___": "CRUE",
    "Fond du ___, Wis.": "LAC",
    "___ Romeo": "ALFA",
    "Kidney-related": "RENAL",
    "Sgt., e.g.": "NCO",
    "Jackie's second": "ARI",
    "Rich soil": "LOAM",
    ""I'll take that as ___"": "ANO",
    "Oil of ___": "OLAY",
    "___-de-sac": "CUL",
    "Actress Swenson": "INGA",
    "Recently": "OFLATE",
    "Prefix with classical": "NEO",
    "Lock of hair": "TRESS",
    "Together, musically": "ADUE",
    "Society page word": "NEE",
    "___ Hari": "MATA",
    "Likely": "APT",
    "Hi-___ monitor": "RES",
    "Bring (out)": "TROT",
    "Ballet bend": "PLIE",
    "Sine ___ non": "QUA",
    "___ King Cole": "NAT",
    "Japanese drama": "NOH",
    ""...___ saw Elba"": "EREI",
    "Author Dinesen": "ISAK",
    "___ contendere": "NOLO",
    "Toledo's lake": "ERIE",
    "Suffix with Capri": "OTE",
    "Dictator Amin": "IDI",
    "Poem of praise": "ODE",
    "___ glance": "ATA",
    "Table scrap": "ORT",
    "___ Rabbit": "BRER",
    "Taj Mahal city": "AGRA",
    "In medias ___": "RES",
    "___ a one": "NARY",
    "Uganda's Amin": "IDI",
    "Birth-related": "NATAL",
    "Spy novelist Deighton": "LEN",
    "___ Bell": "TACO",
    "Opposite of post-": "PRE",
    "Bric-a-___": "BRAC",
    "Maine college town": "ORONO",
    "Caviar": "ROE",
    "___ Dame": "NOTRE",
    ""Lord, is ___?"": "ITI",
    "WNW's opposite": "ESE",
    "Ram's mate": "EWE",
    "Opposite of WSW": "ENE",
    "Art Deco artist": "ERTE",
    "___ Plaines, Ill.": "DES",
    "Jacob's first wife": "LEAH",
    "Actor Wallach": "ELI",
    "Lo-cal": "LITE",
    "Final Four org.": "NCAA",
    "Ye ___ Shoppe": "OLDE",
    "Actress Falco": "EDIE",
    "On ___ with": "APAR",
    "Architect Jones": "INIGO",
    "Actor McGregor": "EWAN",
    ""To Live and Die ___"": "INLA",
    "Part to play": "ROLE",
    "Broadcasting": "ONAIR",
    "Prince Valiant's wife": "ALETA",
    "Actor Jannings": "EMIL",
    "Hardy heroine": "TESS",
    "Reggae relative": "SKA",
    "Lhasa ___ (dog)": "APSO",
    "Mideast carrier": "ELAL",
    "Actor Cariou": "LEN",
    "Trillion: Prefix": "TERA",
    "Greek H's": "ETAS",
    "Italian wine region": "ASTI",
    "Abba of Israel": "EBAN",
    "Mystique": "AURA",
    "Cartoonist Peter": "ARNO",
    "President pro ___": "TEM",
    ""___ Poetica"": "ARS",
    "Surmounting": "ATOP",
    "Author Zora ___ Hurston": "NEALE",
    ""___ of the D'Urbervilles"": "TESS",
    "Actress Russo": "RENE",
    ""___ Maria"": "AVE",
    "Les Etats-___": "UNIS",
    "Bandleader Shaw": "ARTIE",
    "Scottish hillside": "BRAE",
    "Singer India.___": "ARIE",
    "Dadaist Jean": "ARP",
    "Schindler of "Schindler's List"": "OSKAR",
    "Poet Pound": "EZRA",
    "Writer Ephron": "NORA",
    "Tit for ___": "TAT",
    "Buffalo's lake": "ERIE",
    "City near Provo": "OREM",
    "___ alai": "JAI",
    "Prefix with natal": "NEO",
    "Singer Fitzgerald": "ELLA",
    "___ Vegas": "LAS",
    "Fr. holy woman": "STE",
    "Daredevil Knievel": "EVEL",
    "Tide type": "NEAP",
    ""___ the season ..."": "TIS",
    "Lab eggs": "OVA",
    "Iowa college": "COE",
    "Cheer (for)": "ROOT",
    "Kit ___ bar": "KAT",
    "California's Big ___": "SUR",
    "Uno + due": "TRE",
    "Lyricist Gershwin": "IRA",
    "The gamut": "ATOZ",
    "Hit the slopes": "SKI",
    "Wing it": "ADLIB",
    "Change for a five": "ONES",
    "Achy": "SORE",
    "Compete": "VIE",
    "Krazy ___": "KAT",
    ""You've got mail" co.": "AOL",
    "Ave. crossers": "STS",
    ""Waiting for Lefty" playwright": "ODETS",
    "Slightly open": "AJAR",
    "Battery terminal": "ANODE",
    "The "E" of Q.E.D.": "ERAT",
    "French girlfriend": "AMIE",
    "Composer Bruckner": "ANTON",
    "Prefix with potent": "OMNI",
    "Facts and figures": "DATA",
    "March Madness org.": "NCAA",
    "Night school subj.": "ESL",
    "Bete ___": "NOIRE",
    ""This one's ___"": "ONME",
    "Born: Fr.": "NEE",
    "Helper: Abbr.": "ASST",
    "Bro's sibling": "SIS",
    "Cross to bear": "ONUS",
    "Start of North Carolina's motto": "ESSE",
    "In ___ (unborn)": "UTERO",
    "Guitarist Clapton": "ERIC",
    "Actress Zellweger": "RENEE",
    "Not so much": "LESS",
    "Gumbo vegetable": "OKRA",
    "Peculiar: Prefix": "IDIO",
    "Novelist Jaffe": "RONA",
    "Lennon's lady": "ONO",
    "Mild cigar": "CLARO",
    "___ land": "LALA",
    "Book before Nehemiah": "EZRA",
    ""Born Free" lioness": "ELSA",
    "Pianist Claudio": "ARRAU",
    "All over again": "ANEW",
    "Mimicked": "APED",
    "Mountain ridge": "ARETE",
    "Wide shoe spec": "EEE",
    "Taxi": "CAB",
    ""O Sole ___"": "MIO",
    "Splinter group": "SECT",
    "General on Chinese menus": "TSO",
    "___ Solo of "Star Wars"": "HAN",
    ""Well, ___-di-dah!"": "LAH",
    ""Anything ___?"": "ELSE",
    "___ judicata": "RES",
    "Frozen waffle brand": "EGGO",
    "___ Bator, Mongolia": "ULAN",
    "Yours, in Tours": "ATOI",
    "Gaelic tongue": "ERSE",
    "Dressed": "CLAD",
    "Toward shelter": "ALEE",
    "Swiss peak": "ALP",
    "Mayberry boy": "OPIE",
    "Final: Abbr.": "ULT",
    "___ de Janeiro": "RIO",
    "___ Nostra": "COSA",
    "Singer Redding": "OTIS",
    "Singer Lopez": "TRINI",
    "Columnist Bombeck": "ERMA",
    "Chicago trains": "ELS",
    "___' Pea": "SWEE",
    "Photo finish": "MATTE",
    "Dancer Charisse": "CYD",
    "Kemo ___": "SABE",
    "Utah ski resort": "ALTA",
    "General on a Chinese menu": "TSO",
    "Playwright Fugard": "ATHOL",
    "Of the flock": "LAIC",
    "College in New Rochelle, N.Y.": "IONA",
    ""What's ___ for me?"": "INIT",
    "Opposite WSW": "ENE",
    "Greek war god": "ARES",
    "Writer Rand": "AYN",
    "British gun": "STEN",
    ""Garfield" dog": "ODIE",
    "New Rochelle college": "IONA",
    "Italian article": "UNA",
    "Bro or sis": "SIB",
    "___ even keel": "ONAN",
    "Make amends": "ATONE",
    "In case": "LEST",
    "Composer Rorem": "NED",
    "In the manner of": "ALA",
    "Equilibrium": "STASIS",
    "___-ski": "APRES",
    "Kama ___": "SUTRA",
    "Muslim leader": "IMAM",
    "Coeur d'___": "ALENE",
    "___ Arbor, Mich.": "ANN",
    "___ Baba": "ALI",
    "Actress Cannon": "DYAN",
    "Bard's "before"": "ERE",
    ""Illmatic" rapper": "NAS",
    ""Dies ___" (hymn)": "IRAE",
    "NNW's opposite": "SSE",
    "Norm: Abbr.": "STD",
    "Come after": "ENSUE",
    "___ Alto": "PALO",
    "Came up": "AROSE",
    "Shakespearean prince": "HAL",
    "Tokyo, once": "EDO",
    "Mex. miss": "SRTA",
    "Prince Valiant's son": "ARN",
    "___ nova": "BOSSA",
    "___ Friday's": "TGI",
    "Golf peg": "TEE",
    "Is down with": "HAS",
    "Kimono sash": "OBI",
    "___ breve": "ALLA",
    "Similar (to)": "AKIN",
    "Singer Carly ___ Jepsen": "RAE",
    "Singer Brickell": "EDIE",
    "Eagle's nest": "AERIE",
    "Mathematician Turing": "ALAN",
    "___ Pieces": "REESES",
    "Johnson of "Laugh-In"": "ARTE",
    ""___ Rosenkavalier"": "DER",
    "Swenson of "Benson"": "INGA",
    "Rice-A-___": "RONI",
    "Kind of sch.": "ELEM",
    ""Would ___?"": "ILIE",
    "C.I.A. forerunner": "OSS",
    "Author Umberto": "ECO",
    "Author Wiesel": "ELIE",
    "Author Silverstein": "SHEL",
    "Gossipy Barrett": "RONA",
    "Samoan capital": "APIA",
    "Buster Brown's dog": "TIGE",
    "River of Flanders": "YSER",
    "Jazzy Fitzgerald": "ELLA",
    "Vardalos of "My Big Fat Greek Wedding"": "NIA",
    "Curved molding": "OGEE",
    "Plastic ___ Band": "ONO",
    "Georgetown athlete": "HOYA",
    "Folkie Guthrie": "ARLO",
    "Gymnast Comaneci": "NADIA",
    "Not e'en once": "NEER",
    "Author Jong": "ERICA",
    "Mineral suffix": "ITE",
    "Prefix with tourism": "ECO",
    "Prefix with system": "ECO",
    "Nine-digit ID": "SSN",
    "Zhivago's love": "LARA",
    "British verb ending": "ISE",
    "Actress Polo": "TERI",
    "Einstein's birthplace": "ULM",
    "___ Pepper": "SGT",
    ""___ Lang Syne"": "AULD",
    "Rock's Brian": "ENO",
    "Tulsa sch.": "ORU",
    "Gumbo ingredient": "OKRA",
    ""Peter Pan" pirate": "SMEE",
    "El ___, Tex.": "PASO",
    "Egypt's Sadat": "ANWAR",
    "___ du Diable": "ILE",
    "In ___ (actually)": "ESSE",
    ""Beau ___"": "GESTE",
    "Moo ___ pork": "SHU",
    "Zeno's home": "ELEA",
    "As soon as": "ONCE",
    "Elvis ___ Presley": "ARON",
    "Designer Gucci": "ALDO",
    "Salinger girl": "ESME",
    "Ambulance letters": "EMS",
    "Forearm bone": "ULNA",
    ""Bus Stop" playwright": "INGE",
    "Alpha's opposite": "OMEGA",
    "Former Mideast inits.": "UAR",
    "Vogue competitor": "ELLE",
    "Thurman of "Pulp Fiction"": "UMA",
    "Mine, in Marseille": "AMOI",
    "Director Preminger": "OTTO",
    "Pouch": "SAC",
    "French cleric": "ABBE",
    "___ Z": "ATO",
    "Writer Zora ___ Hurston": "NEALE",
    "___ Cruces, N.M.": "LAS",
    "Jai alai basket": "CESTA",
    "Suffix with expert": "ISE",
    "Suffers from": "HAS",
    "___-Locka, Fla.": "OPA",
    "Green: Prefix": "ECO",
    "Oui's opposite": "NON",
    "With: Fr.": "AVEC",
    "___ Dhabi": "ABU",
    "Chow down": "EAT",
    "Artist Chagall": "MARC",
    "___-pitch": "SLO",
    "Magna ___": "CARTA",
    "Madrid Mrs.": "SRA",
    "Thick slice": "SLAB",
    "The Little Mermaid": "ARIEL",
    "Ticked (off)": "TEED",
    "Gladden": "ELATE",
    "New Zealand native": "MAORI",
    "Nick at ___": "NITE",
    "Vogue rival": "ELLE",
    "Voice below soprano": "ALTO",
    "Roof overhang": "EAVE",
    "Formerly, once": "ERST",
    "Director Craven": "WES",
    "Leave in": "STET",
    "___ prof.": "ASST",
    "Mongolian desert": "GOBI",
    "Sign before Virgo": "LEO",
    "Morales of "NYPD Blue"": "ESAI",
    "King Kong, e.g.": "APE",
    "Not kosher": "TREF",
    "___ out a living": "EKE",
    "Prefix with pressure": "ACU",
    "Prefix with puncture": "ACU",
    "Length x width, for a rectangle": "AREA",
    "___ bene": "NOTA",
    "Roe source": "SHAD",
    "Rocky peak": "TOR",
    "Rock's ___ Fighters": "FOO",
    "Density symbol": "RHO",
    "Der ___ (Adenauer)": "ALTE",
    "Novelist Seton": "ANYA",
    "Guy's date": "GAL",
    "Gymnastics coach Karolyi": "BELA",
    "___ Wednesday": "ASH",
    "Peru's capital": "LIMA",
    "Not fer": "AGIN",
    ""___ Miz"": "LES",
    "Disney's "___ and the Detectives"": "EMIL",
    "___ of Sandwich": "EARL",
    "Actor Chaney": "LON",
    "Surrounding glow": "AURA",
    ""Aladdin" prince": "ALI",
    "Riddle-me-___": "REE",
    "Sheltered": "ALEE",
    "Surgery sites, for short": "ORS",
    ""I didn't know that!"": "GEE",
    "Doctors' org.": "AMA",
    ""The ___ the limit!"": "SKYS",
    "Soprano Fleming": "RENEE",
    "Airing": "ONTV",
    "Kind of arch": "OGEE",
    "Scand. land": "NOR",
    "Hypotheticals": "IFS",
    "Bellini opera": "NORMA",
    "Swing around": "SLUE",
    "Muscat native": "OMANI",
    ""Put ___ on it!"": "ALID",
    "Mule of song": "SAL",
    "First place": "EDEN",
    "First-stringers": "ATEAM",
    "___ carte": "ALA",
    "Drop the ball": "ERR",
    "Torah holders": "ARKS",
    "___ noire": "BETE",
    "Upper hand": "EDGE",
    "Phnom ___": "PENH",
    "Philosopher Kierkegaard": "SOREN",
    ""Exodus" author": "URIS",
    "Obliterate": "ERASE",
    "October birthstone": "OPAL",
    "1982 Disney film": "TRON",
    "At full speed": "AMAIN",
    "Canon camera": "EOS",
    "K-12": "ELHI",
    "___ Tome": "SAO",
    "Score after deuce": "ADIN",
    "Ancient Peruvian": "INCA",
    "Actor Quinn": "AIDAN",
    "Related on the mother's side": "ENATE",
    "Actress Graff": "ILENE",
    "___-garde": "AVANT",
    "Harbinger": "OMEN",
    ""2001" computer": "HAL",
    ""Didn't I tell you?"": "SEE",
    "Hatcher of "Lois & Clark"": "TERI",
    ""Mon ___!"": "DIEU",
    "Cubic meter": "STERE",
    "Sports venue": "ARENA",
    "Photographer Goldin": "NAN",
    "Actress Campbell": "NEVE",
    "Puppeteer Lewis": "SHARI",
    "Puppeteer Tony": "SARG",
    "___ Fein": "SINN",
    "Stewpot": "OLLA",
    "Actress McClurg": "EDIE",
    "Actress Merrill": "DINA",
    "Actress Sue ___ Langdon": "ANE",
    "Verdi aria": "ERITU",
    "Crucifix": "ROOD",
    "Jason's ship": "ARGO",
    "Jazz's Fitzgerald": "ELLA",
    "Zaire's Mobutu ___ Seko": "SESE",
    "Broadcaster": "AIRER",
    "Ark builder": "NOAH",
    ""Just kidding!"": "NOT",
    "Timber wolf": "LOBO",
    "Egyptian cobra": "ASP",
    "___ dye": "AZO",
    "Qty.": "AMT",
    "Yalie": "ELI",
    "Writer LeShan": "EDA",
    "End in ___": "ATIE",
    "Thomas ___ Edison": "ALVA",
    "Victorian ___": "ERA",
    "Cross shape": "TAU",
    "Verne captain": "NEMO",
    "Civil War inits.": "CSA",
    "Wildebeest": "GNU",
    "Small dam": "WEIR",
    "___-relief": "BAS",
    "Art Deco designer": "ERTE",
    ""Are you ___ out?"": "INOR",
    "St. Petersburg's river": "NEVA",
    "Corrida cry": "OLE",
    ""___-haw!"": "YEE",
    "Baldwin of "30 Rock"": "ALEC",
    "S-shaped molding": "OGEE",
    "Nouveau ___": "RICHE",
    "Kung ___ chicken": "PAO",
    "Not pro": "ANTI",
    "Capital of Italia": "ROMA",
    "Outpouring": "SPATE",
    "Be up": "BAT",
    "Make up (for)": "ATONE",
    "Entre ___": "NOUS",
    "Entr'___": "ACTE",
    "Mandlikova of tennis": "HANA",
    "Raison ___": "DETRE",
    ""___-hoo!"": "YOO",
    "Slip up": "ERR",
    "Here, to Henri": "ICI",
    "James of jazz": "ETTA",
    "Anise-flavored liqueur": "OUZO",
    "Star in Cygnus": "DENEB",
    "Java neighbor": "BALI",
    "Japanese cartoon art": "ANIME",
    "Sen. Hatch": "ORRIN",
    "Israeli airline": "ELAL",
    "Z ___ zebra": "ASIN",
    "Charon's river": "STYX",
    "German direction": "OST",
    "___ time (never)": "ATNO",
    "Neck of the woods": "AREA",
    "___ were": "ASIT",
    "Garr of "Tootsie"": "TERI",
    "Scratch (out)": "EKE",
    "___ spumante": "ASTI",
    "Scruffs": "NAPES",
    "And so on: Abbr.": "ETC",
    "Scrabble piece": "TILE",
    "Golfer Woosnam": "IAN",
    "Isle of exile": "ELBA",
    "Stick up": "ROB",
    "Scratches (out)": "EKES",
    "Abound": "TEEM",
    "___ soda": "SAL",
    "G.I.'s address": "APO",
    "Muse of history": "CLIO",
    "Up to it": "ABLE",
    "Author Janowitz": "TAMA",
    "Author Jaffe": "RONA",
    "1970 Kinks hit": "LOLA",
    "Motorists' org.": "AAA",
    "Poland's Walesa": "LECH",
    "Sheet music abbr.": "ARR",
    "Sheltered, at sea": "ALEE",
    """

    return most_frequent_clue_answer_pairs
