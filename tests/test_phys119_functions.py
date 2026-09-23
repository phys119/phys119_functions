"""Tests for phys119_functions.

Run with: python -m pytest
"""

import io
import warnings
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

@pytest.mark.parametrize("fn", [pf.mean, pf.standard_deviation, pf.standard_unc_of_mean])
@pytest.mark.parametrize("bad, expected", [
    (5, "single number"),
    ("abc", "list or numpy array of numbers"),
    (["a", "b"], "list or numpy array of numbers"),
    ({"a": 1}, "list or numpy array of numbers"),
    (np.array([[1.0, 2.0], [3.0, 4.0]]), "one-dimensional"),
    ([1.0, float("nan"), 3.0], "missing values"),
    ([1.0, float("inf"), 3.0], "infinite values"),
    ([1.0, float("-inf")], "infinite values"),
    ([], "at least"),
])
def test_bad_data_reports_and_stops(fn, bad, expected):
    error, message = failing_call(fn, bad)
    assert message.startswith("ERROR:")
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


# Non-finite values and calculations that break down

@pytest.mark.parametrize("fn", [pf.standard_deviation, pf.standard_unc_of_mean])
def test_overflow_is_reported_rather_than_returning_inf(fn):
    error, message = failing_call(fn, [1e200, -1e200])
    assert "could not produce a number" in message
    assert fn.__name__ in str(error)


@pytest.mark.parametrize("fn", [pf.standard_deviation, pf.standard_unc_of_mean])
def test_large_but_workable_values_still_compute(fn):
    assert fn([1e150, -1e150]) is not None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_t_score_rejects_non_finite_arguments(bad, position):
    args = [10, 2, 12, 3]
    args[position] = bad
    names = ["A", "dA", "B", "dB"]
    _, message = failing_call(pf.t_score, *args)
    assert "must be an ordinary number" in message
    assert names[position] in message


@pytest.mark.parametrize("fn, args", [
    (pf.standard_deviation, ([1e200, -1e200],)),     # overflows while computing
    (pf.standard_unc_of_mean, ([1e200, -1e200],)),
    (pf.standard_deviation, ([1.0, float("inf")],)),  # rejected before computing
    (pf.t_score, (float("inf"), 2, 12, 3)),
])
def test_no_numpy_warning_reaches_the_student(fn, args):
    """numpy's own RuntimeWarning names a file path, which is what this avoids."""
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        failing_call(fn, *args)
    assert [w for w in raised if issubclass(w.category, RuntimeWarning)] == []


def test_oversized_integer_in_data_is_reported_not_raised_raw():
    """np.asarray raises OverflowError, which must not reach the student."""
    _, message = failing_call(pf.standard_deviation, [10 ** 400, 1])
    assert "too large to work with" in message


def test_oversized_integer_argument_to_t_score():
    _, message = failing_call(pf.t_score, 10 ** 400, 2, 12, 3)
    assert "must be an ordinary number" in message


@pytest.mark.parametrize("fn, args", [
    (pf.t_score, ("x" * 500, 2, 12, 3)),
    (pf.standard_deviation, ([10 ** 400, 1],)),
    (pf.standard_deviation, ("y" * 500,)),
])
def test_messages_stay_short_whatever_was_passed(fn, args):
    _, message = failing_call(fn, *args)
    assert max(len(line) for line in message.splitlines()) < 120


# mean

def test_mean_matches_numpy():
    assert pf.mean(DATA) == pytest.approx(float(np.mean(DATA)))


def test_mean_docstring_examples():
    assert pf.mean(DATA) == pytest.approx(10.07, abs=5e-3)
    assert pf.mean(np.array([1, 2, 3, 4])) == pytest.approx(2.5)


def test_mean_returns_a_plain_float():
    assert type(pf.mean(DATA)) is float


def test_mean_accepts_a_single_measurement():
    """A mean of one number is that number, unlike a spread."""
    assert pf.mean([4.8]) == pytest.approx(4.8)


def test_mean_rejects_an_empty_dataset():
    _, message = failing_call(pf.mean, [])
    assert "at least 1 measurement" in message
    assert "empty list has no mean" in message


@pytest.mark.parametrize("fn", [pf.standard_deviation, pf.standard_unc_of_mean])
def test_spread_functions_still_need_two(fn):
    _, message = failing_call(fn, [4.8])
    assert "at least 2 measurements" in message


@pytest.mark.parametrize("bad, expected", [
    ([1.0, float("nan")], "missing values"),
    ([1.0, float("inf")], "infinite values"),
    (5, "single number"),
    ("abc", "list or numpy array of numbers"),
])
def test_mean_refuses_what_the_others_refuse(bad, expected):
    _, message = failing_call(pf.mean, bad)
    assert expected in message


# Usage help from the decorator

@pytest.mark.parametrize("fn, name", [
    (pf.t_score, "t_score"),
    (pf.mean, "mean"),
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
    assert exported == {"t_score", "mean", "standard_deviation",
                        "standard_unc_of_mean", "Phys119Error", "np"}


def test_star_import_provides_numpy_as_a_backup():
    """A notebook missing its own import numpy as np still gets one."""
    namespace = {}
    exec("from phys119_functions import *", namespace)
    assert namespace["np"] is np


def test_a_student_rebinding_np_cannot_break_the_functions():
    namespace = {}
    exec("from phys119_functions import *", namespace)
    namespace["np"] = 5
    exec("result = mean([1.0, 2.0, 3.0])", namespace)
    assert namespace["result"] == pytest.approx(2.0)


def test_docstrings_do_not_assume_numpy_is_imported():
    """help() must be copyable by a student who imported nothing else."""
    for fn in (pf.t_score, pf.mean, pf.standard_deviation, pf.standard_unc_of_mean):
        assert "np." not in fn.__doc__


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
