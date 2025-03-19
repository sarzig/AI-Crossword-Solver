from clue_classification_and_processing.clue_features import *


def test_uppercase_percentage():
    assert uppercase_percentage("dog") == 0
    assert uppercase_percentage("Dog") == 0
    assert uppercase_percentage("Dog Dog") == 0.5
    assert uppercase_percentage("ALLCAPS!!!") == 0
    assert uppercase_percentage("___ Burgundy, Will Ferrell persona") == 0.6


def test_count_proper_nouns():
    assert count_proper_nouns("John Johnson") == 2
    assert count_proper_nouns("___ Burgundy, Will Ferrell persona") == 3
    assert count_proper_nouns("Eldest son of Cain") == 1