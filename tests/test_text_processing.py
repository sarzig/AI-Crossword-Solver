import pytest

from clue_classification_and_processing.fill_in_the_blank import fill_in_the_blank_with_possible_source, \
    preprocess_lower_remove_punct_strip_whitespace, process_text_into_clue_answer
from puzzle_objects.clue_and_board import Clue


def test_process_text_into_clue_answer():
    assert process_text_into_clue_answer("") == ""
    assert process_text_into_clue_answer("Hey There") == "heythere"
    assert process_text_into_clue_answer("HEYTHERE") == "heythere"
    assert process_text_into_clue_answer("hey there") == "heythere"
    assert process_text_into_clue_answer("\n\nx  ") == "x"
    assert process_text_into_clue_answer('"lo que será, será".') == "loqueserasera"
    assert (process_text_into_clue_answer("áàâäãåāăąȧǎ éèêëēĕėęě. íìîïīĭįıȉȋ óòôöõōŏőȯȱø +úùûüũūŭůűųȕȗ ýÿŷȳɏ ñńņňŉŋ")
            == "aaaaaaaaaaaeeeeeeeeeiiiiiiiiiiooooooooooouuuuuuuuuuuuyyyyynnnnnn")


def test_preprocess_text():
    def assert_preprocess(input_text, expected_output):
        actual_output = preprocess_lower_remove_punct_strip_whitespace(input_text)
        assert actual_output == expected_output, f"\nInput: {input_text}\nExpected: {expected_output}\nActual: {actual_output}"

    test_text1 = """Carney was born in Fort Smith, Northwest Territories, and raised in Edmonton, Alberta. He 
    graduated with a bachelor's degree in economics from Harvard University in 1988, going on to study at the 
    University of Oxford, where he earned a master's degree in 1993 and a doctorate in 1995."""
    result_1 = ("carney was born in fort smith northwest territories and raised in edmonton alberta he graduated with "
                "a bachelors degree in economics from harvard university in 1988 going on to study at the university "
                "of oxford where he earned a masters degree in 1993 and a doctorate in 1995")

    test_text2 = """\n\n\n\t\t\t          """
    result_2 = ""

    test_text3 = """
    
    """

    assert_preprocess(test_text1, result_1)
    assert_preprocess(test_text2, result_2)
    assert_preprocess("john's dog", "johns dog")
    assert_preprocess("honky-tonk", "honkytonk")
    assert_preprocess("ex's", "exs")
    texas_page_text = rf""" The streak included such songs as "Ocean Front Property", "All My Ex's Live in Texas", "Famous Last Words of a Fool", and "Baby Blue". Strait finished the decade by winning the CMA Entertainer of the Year award in 1989. One year later, he won the award again.[31]"""
    texas_result = 'the streak included such songs as ocean front property all my exs live in '\
                   'texas famous last words of a fool and baby blue strait finished the decade '\
                   'by winning the cma entertainer of the year award in 1989 one year later he '\
                   'won the award again 31'
    assert_preprocess(texas_page_text, texas_result)


def test_fill_in_the_blank_with_possible_source():
    clue_text = '"I could a tale unfold ___ lightest word / Would harrow up thy soul ...": "Hamlet"'
    possible_source_text = """“I could a tale unfold whose lightest word
    Would harrow up thy soul, freeze thy young blood,
    Make thy two eyes like stars start from their spheres,
    Thy knotted and combined locks to part,
    And each particular hair to stand on end
    Like quills upon the fretful porpentine.
    But this eternal blazon must not be
    To ears of flesh and blood.
    List, list, O list!”
    """

    possible_source_short = "I could a tale unfold whose lightest word Would harrow up thy soul"

    clue_text_blank_at_beginning = '"___ lightest word / Would harrow up thy soul ...": "Hamlet"'
    clue_text_blank_at_end = '"I could a tale unfold whose lightest word / Would harrow up thy ___ ...": "Hamlet"'
    clue_text_blank_at_end2 = '"I could a tale unfold whose lightest word / Would harrow up thy ___": "Hamlet"'

    clue_exes_in_texas = 'George Strait\'s "All My ___ Live in Texas"'
    texas_page_text = rf""" The streak included such songs as "Ocean Front Property", "All My Ex's Live in Texas", "Famous Last Words of a Fool", and "Baby Blue". Strait finished the decade by winning the CMA Entertainer of the Year award in 1989. One year later, he won the award again.[31]"""

    assert fill_in_the_blank_with_possible_source(Clue(clue_text),
                                                  possible_source_text) == "whose"
    assert fill_in_the_blank_with_possible_source(Clue(clue_text),
                                                  possible_source_short) == "whose"
    assert fill_in_the_blank_with_possible_source(Clue(clue_text_blank_at_beginning),
                                                  possible_source_text) == "whose"
    assert fill_in_the_blank_with_possible_source(Clue(clue_text_blank_at_end),
                                                  possible_source_text) == "soul"
    assert fill_in_the_blank_with_possible_source(Clue(clue_text_blank_at_end2),
                                                  possible_source_text) == "soul"
    assert fill_in_the_blank_with_possible_source(Clue('the quote goes as: "hello ___ lady"'),
                                                  "hello pretty lady") == "pretty"
    assert fill_in_the_blank_with_possible_source(Clue('the quote goes as: "hello ___ ___ ___ lady"'),
                                                  "hello pretty pretty pretty lady") == "pretty"
    assert fill_in_the_blank_with_possible_source(Clue(clue_exes_in_texas),
                                                  texas_page_text) == "exs"

    #everly = ('f 1957. Additional hits, including "Wake Up Little Susie," "All I Have to Do Is Dream," and "Problems", '
    #          'would follow through 1958. In')
    #assert fill_in_the_blank_with_possible_source(Clue('test "all I have to ___ Dream"'), everly) == "do is"
