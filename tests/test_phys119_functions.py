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
    assert pf.t_score(dx2=3, x2=12, dx1=2, x1=10) == pytest.approx(0.5547001962252291)


def test_standard_deviation_matches_numpy():
    assert pf.standard_deviation(DATA) == pytest.approx(np.std(DATA, ddof=1))


def test_standard_unc_of_mean_matches_numpy():
    expected = np.std(DATA, ddof=1) / np.sqrt(len(DATA))
    assert pf.standard_unc_of_mean(DATA) == pytest.approx(expected)


def test_docstring_examples():
    """Exactly the calls the docstrings show, so help() cannot go stale."""
    assert pf.standard_deviation(DATA) == pytest.approx(0.2517, abs=5e-5)
    assert pf.standard_unc_of_mean(DATA) == pytest.approx(0.1453, abs=5e-5)
    assert pf.standard_deviation([1, 2, 3, 4]) == pytest.approx(1.291, abs=5e-4)
    assert pf.standard_unc_of_mean([1, 2, 3, 4]) == pytest.approx(0.6455, abs=5e-5)


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
    (np.array([[1.0, 2.0], [3.0, 4.0]]), "one set of measurements"),
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
    assert "needs x1 to be a number" in message


def test_t_score_rejects_two_zero_uncertainties():
    _, message = failing_call(pf.t_score, 10, 0, 12, 0)
    assert "both zero" in message


def test_t_score_argument_names_are_the_ones_students_see():
    """The names in the messages match the notation used in the course."""
    import inspect
    assert list(inspect.signature(pf.t_score).parameters) == ["x1", "dx1", "x2", "dx2"]


# Warnings: message printed, result still returned

def test_t_score_warns_on_negative_uncertainty_but_still_returns():
    result, message = call(pf.t_score, 10, -2, 12, 3)
    assert result == pytest.approx(0.5547001962252291)
    assert "uncertainties to be non-negative" in message


def test_t_score_warns_on_large_uncertainty_but_still_returns():
    result, message = call(pf.t_score, 1, 5, 12, 3)
    assert result is not None
    assert "look large relative" in message


# Warnings from the data functions

TYPO = [439.3, 431.6, 434.6, 4336.0, 439.3, 442.6, 428.6]   # 4336 for 433.6
FLAT = [2.31, 2.31, 2.31, 2.31, 2.31, 2.31]                 # instrument too coarse


@pytest.mark.parametrize("fn", [pf.mean, pf.standard_deviation, pf.standard_unc_of_mean])
def test_one_value_far_from_the_rest_warns_but_still_returns(fn):
    result, message = call(fn, TYPO)
    assert result is not None
    assert "far from the others" in message
    assert "4336 at position 3" in message
    assert "428.6 and 442.6" in message
    assert f"{fn.__name__}()" in message


@pytest.mark.parametrize("fn", [pf.standard_deviation, pf.standard_unc_of_mean])
def test_identical_measurements_warn_but_still_return_zero(fn):
    result, message = call(fn, FLAT)
    assert result == pytest.approx(0.0)
    assert "spread is exactly zero" in message
    assert "all 6 measurements = 2.31" in message


def test_mean_says_nothing_about_spread():
    """A mean of identical values is unremarkable; only the spread functions warn."""
    result, message = call(pf.mean, FLAT)
    assert result == pytest.approx(2.31)
    assert message == ""


# The false positives that would matter most

@pytest.mark.parametrize("fn", [pf.mean, pf.standard_deviation, pf.standard_unc_of_mean])
@pytest.mark.parametrize("data", [
    [439.3, 431.6, 434.6, 433.3, 439.3, 442.6, 428.6, 441.6],  # a real lab set
    [2.31, 2.34, 2.29, 2.33, 2.30],                            # tight repeats
    [1.0, 2.0, 3.0, 4.0, 5.0],                                 # evenly spread
    [10.0, 10.1],                                              # too few to judge
    [5.0, 5.1, 5.2],                                           # still too few
])
def test_ordinary_data_produces_no_warning(fn, data):
    result, message = call(fn, data)
    assert result is not None
    assert message == ""


def test_the_outlier_check_needs_four_values():
    """With three points there is no 'the others' to compare against."""
    result, message = call(pf.mean, [1.0, 1.1, 99.0])
    assert result is not None
    assert message == ""


def test_an_outlier_cannot_hide_behind_its_own_size():
    """A mean-and-standard-deviation test would miss this; the median does not."""
    result, message = call(pf.standard_deviation, [5.0, 5.1, 5.2, 5.05, 5000.0])
    assert "far from the others" in message
    assert "5000 at position 4" in message


# Non-finite values and calculations that break down

@pytest.mark.parametrize("fn", [pf.mean, pf.standard_deviation, pf.standard_unc_of_mean])
def test_overflow_is_reported_rather_than_returning_inf(fn):
    """Summing these overflows a float, whichever of the three is asked."""
    error, message = failing_call(fn, [1e308, 1e308])
    assert "could not produce a number" in message
    assert fn.__name__ in str(error)


@pytest.mark.parametrize("fn", [pf.standard_deviation, pf.standard_unc_of_mean])
def test_overflow_in_the_squared_deviations_is_also_reported(fn):
    """These two square the deviations, so they overflow where a mean does not."""
    _, message = failing_call(fn, [1e200, -1e200])
    assert "could not produce a number" in message


def test_mean_survives_values_that_overflow_a_spread():
    """The same data a spread cannot handle still has a perfectly good mean."""
    assert pf.mean([1e200, -1e200]) == pytest.approx(0.0)


@pytest.mark.parametrize("fn", [pf.mean, pf.standard_deviation, pf.standard_unc_of_mean])
def test_large_but_workable_values_still_compute(fn):
    assert fn([1e150, -1e150]) is not None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_t_score_rejects_non_finite_arguments(bad, position):
    args = [10, 2, 12, 3]
    args[position] = bad
    names = ["x1", "dx1", "x2", "dx2"]
    _, message = failing_call(pf.t_score, *args)
    assert "to be an ordinary number" in message
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
    assert "to be an ordinary number" in message


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
    assert pf.mean([1, 2, 3, 4]) == pytest.approx(2.5)


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


# Every message says which function produced it

@pytest.mark.parametrize("fn, args", [
    (pf.t_score, ("ten", 2, 12, 3)),
    (pf.t_score, (float("nan"), 2, 12, 3)),
    (pf.t_score, (10, 0, 12, 0)),
    (pf.t_score, (10, 2, 12)),
    (pf.mean, ([],)),
    (pf.mean, (4.8,)),
    (pf.mean, ("abc",)),
    (pf.mean, ([1.0, float("nan")],)),
    (pf.mean, ([1.0, float("inf")],)),
    (pf.mean, ([1e308, 1e308],)),
    (pf.mean, ([10 ** 400],)),
    (pf.standard_deviation, ([4.8],)),
    (pf.standard_deviation, (np.array([[1.0, 2.0], [3.0, 4.0]]),)),
    (pf.standard_unc_of_mean, ([4.8],)),
    (pf.standard_unc_of_mean, ([],)),
])
def test_every_error_names_the_function_that_produced_it(fn, args):
    """A cell may call several of these, so a message must identify itself."""
    error, message = failing_call(fn, *args)
    assert f"{fn.__name__}()" in message.splitlines()[0]
    assert f"{fn.__name__}()" in str(error)


@pytest.mark.parametrize("args", [
    (9.81, 0.05, 9.79, -0.04),     # negative uncertainty
    (9.81, 12.0, 9.79, 0.05),      # uncertainty larger than the measurement
])
def test_every_warning_names_the_function_that_produced_it(args):
    result, message = call(pf.t_score, *args)
    assert result is not None
    assert "t_score()" in message.splitlines()[0]


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


def test_messages_use_jupyters_own_machine_output_colours():
    """Pins the 1.2.0 decision: a message must not look like a course callout."""
    assert "--jp-rendermime-error-background" in pf._ERROR_STYLE["background"]
    assert "--jp-warn-color3" in pf._WARNING_STYLE["background"]
    assert "--jp-content-font-color1" in pf._ERROR_STYLE["colour"]


def test_message_boxes_are_flat_and_square():
    """Course callouts are rounded with a left bar; these must not be."""
    assert "border" not in pf._BOX
    assert "border-radius" not in pf._BOX


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
