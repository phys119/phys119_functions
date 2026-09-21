"""Helper functions for Phys 119 lab notebooks.

Developed by Joss Ives for UBC Physics Labs, in collaboration with Claude
(Anthropic), 2026. The student-facing error messages, the input checking and
the usage help came out of that collaboration.

standard_deviation and standard_unc_of_mean are adapted from functions
developed by Rebeckah Fussell for Cornell Physics Labs.

Students import everything at the top of a lab notebook:

    from phys119_functions import *

and get help on any function with, for example, help(t_score).
"""

import functools
import sys
import inspect

import numpy as np

__version__ = "1.0.0"

__all__ = [
    "t_score",
    "standard_deviation",
    "standard_unc_of_mean",
]


def _friendly_errors(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except TypeError as e:
            params = [
                (name, p) for name, p in inspect.signature(fn).parameters.items()
                if not name.startswith('_')
            ]
            def _param_str(name, p):
                return name if p.default is inspect.Parameter.empty else f"{name}={p.default!r}"
            required = [name for name, p in params if p.default is inspect.Parameter.empty]
            required_sig = f"({', '.join(required)})"
            full_sig = f"({', '.join(_param_str(n, p) for n, p in params)})"
            if required_sig == full_sig:
                usage = f"Usage:          {fn.__name__}{required_sig}"
            else:
                usage = (
                    f"Usage (required): {fn.__name__}{required_sig}\n"
                    f"Usage (full):     {fn.__name__}{full_sig}"
                )
            print(
                f"Error: {e}\n"
                f"\n"
                f"{usage}\n"
                f"Run help({fn.__name__}) for details.",
                file=sys.stderr,
            )
    return wrapper


@_friendly_errors
def t_score(A, dA, B, dB):
    """Returns the t'-score comparing two measurements.

    Arguments (positional order: A, dA, B, dB):
      A  -- measurement A
      dA -- uncertainty of measurement A
      B  -- measurement B
      dB -- uncertainty of measurement B

    Can be called positionally or with named arguments in any order:
      t_score(10, 2, 12, 3)             ->  0.5547
      t_score(A=10, dA=2, B=12, dB=3)   ->  0.5547
      t_score(A=10, B=12, dA=2, dB=3)   ->  0.5547
    """
    for name, val in [('A', A), ('dA', dA), ('B', B), ('dB', dB)]:
        if not isinstance(val, (int, float, np.number)):
            print(
                f"Error: {name} must be a number, got {type(val).__name__} {val!r}.\n"
                "  Argument order: t_score(A, dA, B, dB)\n"
                "  Example:        t_score(10, 2, 12, 3)  ->  0.5547",
                file=sys.stderr,
            )
            return

    if dA < 0 or dB < 0:
        _w = max(len(str(v)) for v in [A, dA, B, dB])
        def _row(name, val, highlight=False):
            marker = "   !!! must be >= 0 !!!" if highlight else ""
            return f"  {name:<2} = {str(val):>{_w}}{marker}"
        print(
            "Warning: uncertainties must be non-negative.\n"
            + _row('A',  A) + "\n"
            + _row('dA', dA, highlight=dA < 0) + "\n"
            + _row('B',  B) + "\n"
            + _row('dB', dB, highlight=dB < 0),
            file=sys.stderr,
        )

    if abs(dA) >= abs(A) or abs(dB) >= abs(B):
        _w = max(len(str(v)) for v in [A, dA, B, dB])
        def _row(name, val, highlight=False):
            marker = "   !!! unexpectedly large !!!" if highlight else ""
            return f"  {name:<2} = {str(val):>{_w}}{marker}"
        print(
            "Warning: one or more uncertainties look large relative to their measurement.\n"
            + _row('A',  A) + "\n"
            + _row('dA', dA, highlight=abs(dA) >= abs(A)) + "\n"
            + _row('B',  B) + "\n"
            + _row('dB', dB, highlight=abs(dB) >= abs(B)) + "\n"
            + "\n"
            + "Expected argument order: t_score(A, dA, B, dB)\n"
            + "If your values are correct, you can ignore this warning.",
            file=sys.stderr,
        )

    if dA == 0 and dB == 0:
        print(
            "Error: dA and dB cannot both be zero (division by zero).\n"
            "\n"
            "Check that you have entered your uncertainties correctly.",
            file=sys.stderr,
        )
        return

    return float(abs(A - B) / np.sqrt(dA**2 + dB**2))


def _check_1d_data(data, fn_name):
    """Validate a one-dimensional numeric dataset.

    Returns the data as a float numpy array, or prints a student-facing
    error message to stderr and returns None.
    """
    example = f"  Example: {fn_name}([10.1, 10.3, 9.8])"

    def _shown(val):
        text = repr(val)
        return text if len(text) <= 60 else text[:57] + "..."

    if isinstance(data, (int, float, np.number)):
        print(
            "Error: data must be a list or numpy array of measurements, "
            f"got the single number {_shown(data)}.\n"
            f"{example}",
            file=sys.stderr,
        )
        return None

    try:
        arr = np.asarray(data, dtype=float)
    except (TypeError, ValueError):
        print(
            "Error: data must be a list or numpy array of numbers, "
            f"got {type(data).__name__} {_shown(data)}.\n"
            f"{example}",
            file=sys.stderr,
        )
        return None

    if arr.ndim != 1:
        print(
            f"Error: data must be one-dimensional, got an array with shape {arr.shape}.\n"
            "  Pass one set of repeated measurements at a time.\n"
            f"{example}",
            file=sys.stderr,
        )
        return None

    if np.any(np.isnan(arr)):
        print(
            "Error: data contains missing values (nan).\n"
            f"  nan values at indices: {np.where(np.isnan(arr))[0].tolist()}\n"
            "\n"
            "Blank cells in a spreadsheet are read in as nan. Fill them in or\n"
            "remove them from the array, then run this cell again.",
            file=sys.stderr,
        )
        return None

    if arr.size < 2:
        print(
            f"Error: {fn_name} needs at least 2 measurements, got {arr.size}.\n"
            "\n"
            "The spread of a set of measurements cannot be found from fewer\n"
            "than two of them. Check that you passed the full array of\n"
            "repeated measurements.",
            file=sys.stderr,
        )
        return None

    return arr


def _sample_std(arr):
    """Sample standard deviation (N-1 in the denominator) of a 1-D array."""
    N = len(arr)
    return np.sqrt(np.sum((arr - np.mean(arr))**2) / (N - 1))


@_friendly_errors
def standard_deviation(data):
    """Returns the standard deviation of a set of repeated measurements.

    Arguments:
      data -- a list or numpy array of measurements (at least 2 values)

    This is the sample standard deviation: the sum of the squared deviations
    from the mean is divided by N - 1, where N is the number of measurements.
    It describes the spread of the individual measurements.

    Examples:
      standard_deviation([10.1, 10.3, 9.8])        ->  0.2517
      standard_deviation(np.array([1, 2, 3, 4]))   ->  1.291

    Credit: adapted from a function developed by Rebeckah Fussell for
    Cornell Physics Labs.
    """
    arr = _check_1d_data(data, 'standard_deviation')
    if arr is None:
        return

    return float(_sample_std(arr))


@_friendly_errors
def standard_unc_of_mean(data):
    """Returns the standard uncertainty of the mean of repeated measurements.

    Arguments:
      data -- a list or numpy array of measurements (at least 2 values)

    This is the standard deviation divided by the square root of the number
    of measurements N. It describes how well the mean itself is known, so
    unlike the standard deviation it gets smaller as more measurements are
    added.

    Examples:
      standard_unc_of_mean([10.1, 10.3, 9.8])      ->  0.1453
      standard_unc_of_mean(np.array([1, 2, 3, 4])) ->  0.6455

    Credit: adapted from a function developed by Rebeckah Fussell for
    Cornell Physics Labs.
    """
    arr = _check_1d_data(data, 'standard_unc_of_mean')
    if arr is None:
        return

    N = len(arr)
    return float(_sample_std(arr) / np.sqrt(N))
