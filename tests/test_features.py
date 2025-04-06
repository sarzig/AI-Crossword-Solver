from clue_classification_and_processing.clue_features import *
from test_text_processing import float_equal


def test_uppercase_percentage():
    def assert_with_print(actual, expected, clue):
        if isinstance(expected, float):
            assert float_equal(actual, expected), f"FAILED: uppercase_percentage('{clue}') → Expected {expected}, but got {actual}"
        else:
            assert actual == expected, f"FAILED: uppercase_percentage('{clue}') → Expected {expected}, but got {actual}"

    assert_with_print(uppercase_percentage("dog"), 0, "dog")
    assert_with_print(uppercase_percentage("Dog"), 0, "Dog")
    assert_with_print(uppercase_percentage("Dog Dog"), 0.5, "Dog Dog")
    assert_with_print(uppercase_percentage("ALLCAPS!!!"), 1, "ALLCAPS!!!")
    assert_with_print(uppercase_percentage("___ Burgundy, Will Ferrell persona"), 3/5, "___ Burgundy, Will Ferrell persona")
    assert_with_print(uppercase_percentage("Action done while saying 'Good dog'"), 2/6, "Action done while saying 'Good dog'")
    assert_with_print(uppercase_percentage("dog eat Cat"), .333, "dog eat Cat")

    # Some harder ones which rely on POS tagging
    assert_with_print(uppercase_percentage('Bassett of "Black Panther"'), 3/4, 'Bassett of "Black Panther"')
    assert_with_print(uppercase_percentage('Actress Moreno'), .5, 'Actress Moreno')
    assert_with_print(uppercase_percentage('Vigorous exercise'), 0, 'Vigorous exercise')
    assert_with_print(uppercase_percentage('Eyelid affliction'), 0, 'Eyelid affliction')
    assert_with_print(uppercase_percentage('Director Spike'), .5, 'Director Spike')

    print("✅ test_uppercase_percentage PASSED")


def test_count_proper_nouns():
    assert count_proper_nouns("John Johnson") == 2
    assert count_proper_nouns("___ Burgundy, Will Ferrell persona") == 3
    assert count_proper_nouns("Eldest son of Cain") == 1