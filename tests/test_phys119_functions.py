"""Tests for phys119_functions.

Run with: python -m pytest
"""

import io
from contextlib import redirect_stderr

import numpy as np
import pytest

import phys119_functions as pf


def call(fn, *args, **kwargs):
    """Call fn, returning (result, text printed to stderr)."""
    buffer = io.StringIO()
    with redirect_stderr(buffer):
        result = fn(*args, **kwargs)
    return result, buffer.getvalue()


def failing_call(fn, *args, **kwargs):
    """Call fn expecting it to stop, returning (exception, text on stderr)."""
    buffer = io.StringIO()
    with redirect_stderr(buffer):
        with pytest.raises(pf.Phys119Error) as caught:
            fn(*args, **kwargs)
    return caught.value, buffer.getvalue()


DATA = [10.1, 10.3, 9.8]


# Values

def test_t_score_value():
    assert pf.t_score(10, 2, 12, 3) == pytest.approx(0.5547001962252291)


def test_t_score_named_arguments_in_any_order():
    assert pf.t_score(dB=3, B=12, dA=2, A=10) == pytest.approx(0.5547001962252291)


def test_standard_deviation_matches_numpy():
    assert pf.standard_deviation(DATA) == pytest.approx(np.std(DATA, ddof=1))


def test_standard_unc_of_mean_matches_numpy():
    expected = np.std(DATA, ddof=1) / np.sqrt(len(DATA))
    assert pf.standard_unc_of_mean(DATA) == pytest.approx(expected)


def test_docstring_examples():
    assert pf.standard_deviation(DATA) == pytest.approx(0.2517, abs=5e-5)
    assert pf.standard_unc_of_mean(DATA) == pytest.approx(0.1453, abs=5e-5)
    assert pf.standard_deviation(np.array([1, 2, 3, 4])) == pytest.approx(1.291, abs=5e-4)
    assert pf.standard_unc_of_mean(np.array([1, 2, 3, 4])) == pytest.approx(0.6455, abs=5e-5)


def test_results_are_plain_floats():
    assert type(pf.t_score(10, 2, 12, 3)) is float
    assert type(pf.standard_deviation(DATA)) is float
    assert type(pf.standard_unc_of_mean(DATA)) is float


@pytest.mark.parametrize("data", [DATA, tuple(DATA), np.array(DATA), np.array([1, 2, 3, 4])])
def test_accepts_sequence_types(data):
    assert pf.standard_deviation(data) is not None


# Errors: message shown in red, execution stopped

@pytest.mark.parametrize("fn", [pf.standard_deviation, pf.standard_unc_of_mean])
@pytest.mark.parametrize("bad, expected", [
    (5, "single number"),
    ("abc", "list or numpy array of numbers"),
    (["a", "b"], "list or numpy array of numbers"),
    ({"a": 1}, "list or numpy array of numbers"),
    (np.array([[1.0, 2.0], [3.0, 4.0]]), "one-dimensional"),
    ([1.0, float("nan"), 3.0], "missing values"),
    ([4.2], "at least 2 measurements"),
    ([], "at least 2 measurements"),
])
def test_bad_data_reports_and_stops(fn, bad, expected):
    error, message = failing_call(fn, bad)
    assert message.startswith("Error:")
    assert expected in message
    assert expected in str(error)


def test_nan_message_lists_indices():
    _, message = failing_call(pf.standard_deviation, [1.0, float("nan"), 3.0, float("nan")])
    assert "[1, 3]" in message


def test_t_score_rejects_non_numbers():
    _, message = failing_call(pf.t_score, "ten", 2, 12, 3)
    assert "A must be a number" in message


def test_t_score_rejects_two_zero_uncertainties():
    _, message = failing_call(pf.t_score, 10, 0, 12, 0)
    assert "cannot both be zero" in message


# Warnings: message printed, result still returned

def test_t_score_warns_on_negative_uncertainty_but_still_returns():
    result, message = call(pf.t_score, 10, -2, 12, 3)
    assert result == pytest.approx(0.5547001962252291)
    assert "must be non-negative" in message


def test_t_score_warns_on_large_uncertainty_but_still_returns():
    result, message = call(pf.t_score, 1, 5, 12, 3)
    assert result is not None
    assert "look large relative" in message


# Usage help from the decorator

@pytest.mark.parametrize("fn, name", [
    (pf.t_score, "t_score"),
    (pf.standard_deviation, "standard_deviation"),
    (pf.standard_unc_of_mean, "standard_unc_of_mean"),
])
def test_missing_arguments_print_usage(fn, name):
    _, message = failing_call(fn)
    assert "Usage:" in message
    assert name in message
    assert "help(" + name + ")" in message


def test_too_many_arguments_print_usage():
    _, message = failing_call(pf.standard_deviation, DATA, DATA)
    assert "Usage:" in message


# Packaging

def test_star_import_exports_only_the_public_api():
    namespace = {}
    exec("from phys119_functions import *", namespace)
    exported = {n for n in namespace if not n.startswith("__")}
    assert exported == {"t_score", "standard_deviation", "standard_unc_of_mean",
                        "Phys119Error"}


def test_version_matches_pyproject():
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    declared = re.search(r'^version = "([^"]+)"', text, re.MULTILINE).group(1)
    assert pf.__version__ == declared


def test_help_text_mentions_credit():
    for fn in (pf.standard_deviation, pf.standard_unc_of_mean):
        assert "Cornell Physics Labs" in fn.__doc__
