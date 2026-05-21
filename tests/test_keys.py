import pytest

from modulation_key_estimator.keys import key_name, key_to_index, normalize_key, shortest_semitone_shift


def test_normalize_key_accepts_flats_and_symbols():
    assert normalize_key("Db") == "c#"
    assert normalize_key("E♭") == "d#"
    assert normalize_key("F♯") == "f#"


def test_key_to_index_and_key_name_round_trip():
    assert key_to_index("a#") == 10
    assert key_name(10) == "a#"


def test_shortest_semitone_shift_wraps():
    assert shortest_semitone_shift(11, 0) == 1
    assert shortest_semitone_shift(0, 11) == -1


def test_invalid_key_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown key"):
        normalize_key("h")

