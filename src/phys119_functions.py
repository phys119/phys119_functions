"""Helper functions for Phys 119 lab notebooks.

The measurement helpers (t_score, mean, standard_deviation,
standard_unc_of_mean) and the plotting functions (plot_data,
plot_residuals, and the interactive autofit and manual_fit). Check
`phys119_functions.__version__` to see which version is in use.

Dropping this file beside a notebook is enough to try everything out: Python
imports a module from the notebook's own folder before any installed one, so
nothing needs installing. It needs numpy, matplotlib for the plotting
functions, and anywidget for autofit and manual_fit.

Developed by Joss Ives for UBC Physics Labs, in collaboration with Claude
(Anthropic), 2026. The whole module, including the student-facing error and
warning messages, the plotting behaviour and the input checking, came out of
that collaboration.

standard_deviation and standard_unc_of_mean are adapted from functions
developed by Rebeckah Fussell for Cornell Physics Labs.

`from phys119_functions import *` also brings in `np`, as a backup for a
notebook whose own `import numpy as np` is missing. Lab notebooks should
still carry that import: it is what students are taught, and it is what makes
their code work anywhere else. The two names refer to the same module, so
importing numpy as well costs nothing, and a student who rebinds `np` in
their own notebook cannot affect the functions here.

How mistakes are reported:

- A warning appears as an amber box. The calculation carries on and the
  result is still returned, because the values might be right.
- An error appears as a red box and stops the cell by raising Phys119Error.
  Nothing further down the notebook then runs on a result that was never
  produced. In a notebook no traceback is shown, only the box.
- Outside a notebook, both fall back to plain text on stderr and the
  exception behaves like any other Python exception.
"""

import ast
import functools
import html
import sys
import inspect
import json
import linecache
import os
import re

import numpy as np
import matplotlib.pyplot as plt

# Kept in step with the installed package, bumped together. A ".dev0"
# suffix marks shared code that is ahead of the deployed package.
__version__ = "1.5.0"

__all__ = [
    "t_score",
    "mean",
    "standard_deviation",
    "standard_unc_of_mean",
    "plot_data",
    "plot_residuals",
    "autofit",
    "manual_fit",
    "Phys119Error",
    "np",
]


class Phys119Error(Exception):
    """Raised when a phys119 function is called in a way that cannot work.

    The explanation a student needs is shown as a red box above this. The
    exception itself is what stops the cell, so that nothing further down the
    notebook runs on a result that was never produced.
    """


# Both boxes borrow Jupyter's own variables, so a message sits in the same
# colours the notebook already uses for machine output, in any theme. The
# fallbacks are the light-theme values, for anywhere those variables are
# undefined.
_ERROR_STYLE = {
    'background': 'var(--jp-rendermime-error-background, #FDD)',
    'colour': 'var(--jp-content-font-color1, #1A1A1A)',
}

_WARNING_STYLE = {
    'background': 'var(--jp-warn-color3, #FFE0B2)',
    'colour': 'var(--jp-content-font-color1, #1A1A1A)',
}

# Flat, square cornered and unbordered, which is how Jupyter draws a
# traceback. Authored callout boxes in the course notebooks use a pale fill
# with a rounded left bar, so the two no longer read as the same thing.
# The ignore classes stop MathJax (3 in JupyterLab, 2 in HTML exports) from
# typesetting a message's $...$ pairs, such as "$F$" in a LaTeX hint, as maths.
_BOX = (
    '<div class="mathjax_ignore tex2jax_ignore"'
    ' style="background:{background}; color:{colour};'
    ' padding: 8px 12px; margin: 2px 0;'
    ' font-family: var(--jp-code-font-family, monospace); font-size: 13px;'
    ' line-height: 1.45; white-space: pre-wrap;">'
    '<b>{title}:</b> {body}</div>'
)


def _notebook_display():
    """Return (display, HTML) when running in a notebook, else None."""
    try:
        from IPython import get_ipython
        from IPython.display import display, HTML
    except ImportError:
        return None
    shell = get_ipython()
    if shell is None or not hasattr(shell, 'kernel'):
        return None
    return display, HTML


def _show(title, message, style):
    """Show a student-facing message, boxed in a notebook and plain elsewhere."""
    lines = message.split('\n')
    pair = _notebook_display()
    if pair is None:
        print(f"{title}: {lines[0]}", file=sys.stderr)
        for line in lines[1:]:
            print(line, file=sys.stderr)
        return
    display, HTML = pair
    body = '<br>'.join(html.escape(line) if line else '&nbsp;' for line in lines)
    display(HTML(_BOX.format(title=title, body=body, **style)))


def _named(fn_name, message):
    """Make sure a message says which function produced it.

    Messages are written with the function as the subject, as in "mean() needs
    at least 1 measurement". Anything that does not already start that way gets
    the name put in front, so no message can reach a student unattributed. It
    matters because a single cell may call several of these functions.
    """
    return message if message.startswith(f"{fn_name}(") else f"{fn_name}(): {message}"


def _warn(fn_name, message):
    """Show a warning. The calculation carries on and still returns a result."""
    _show('Warning', _named(fn_name, message), _WARNING_STYLE)


def _fail(fn_name, message):
    """Show an error and stop the cell, so nothing downstream uses a bad result."""
    message = _named(fn_name, message)
    _show('ERROR', message, _ERROR_STYLE)
    lines = message.split('\n')
    summary = lines[0]
    for line in lines[1:]:
        if line.startswith('  got '):
            summary = f"{summary} {line.strip()}"
            break
    raise Phys119Error(summary) from None


def _shown(value):
    """A short repr, so a huge value cannot flood the message."""
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _as_finite_number(value):
    """Return value as a float, or None if it is not an ordinary number."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _finite_result(value, fn_name):
    """Return value as a plain float, or stop if the calculation broke down."""
    if not np.isfinite(value):
        _fail(fn_name,
            f"{fn_name}() could not produce a number from these measurements.\n"
            "\n"
            "The calculation overflowed, which usually means one or more values\n"
            "are far larger than they should be. Check the measurements you\n"
            "passed in."
        )
    return float(value)


def _install_traceback_suppressor():
    """Show only the red box for Phys119Error, never a traceback."""
    try:
        from IPython import get_ipython
    except ImportError:
        return
    shell = get_ipython()
    if shell is None or not hasattr(shell, 'set_custom_exc'):
        return

    def _handler(shell_self, etype, value, tb, tb_offset=None):
        return []

    # IPython keeps a single registration, so another copy of this module
    # imported earlier (the installed package beside a development copy, say)
    # would otherwise lose its registration and show full tracebacks again.
    # Keep any Phys119Error already registered, and add this one.
    earlier = tuple(
        cls for cls in (getattr(shell, 'custom_exceptions', None) or ())
        if getattr(cls, '__name__', '') == 'Phys119Error' and cls is not Phys119Error
    )
    shell.set_custom_exc(earlier + (Phys119Error,), _handler)


_install_traceback_suppressor()


def _friendly_errors(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except TypeError as e:
            # A TypeError can come from the call itself (a missing or extra
            # argument) or from inside the function (a value of the wrong
            # kind). Only the first is a usage mistake. The second used to
            # get the usage message too, which told a student whose call was
            # correct that it was not.
            try:
                inspect.signature(fn).bind(*args, **kwargs)
            except TypeError:
                pass
            else:
                _fail(fn.__name__,
                    f"{fn.__name__}() could not work with the values it was given.\n"
                    f"  Python reported: {e}\n"
                    f"\n"
                    f"Check that each argument holds what help({fn.__name__})\n"
                    f"describes, then run this cell again."
                )
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
            # Python's own wording already opens with the function name, in
            # the same form these messages use, so it is left as it is.
            _fail(fn.__name__,
                f"{e}\n"
                f"\n"
                f"{usage}\n"
                f"Run help({fn.__name__}) for details."
            )
    # help() shows the signature; leave out parameters meant only for use
    # inside this module, as the usage message above does.
    signature = inspect.signature(fn)
    wrapper.__signature__ = signature.replace(parameters=[
        p for name, p in signature.parameters.items() if not name.startswith('_')
    ])
    return wrapper


@_friendly_errors
def t_score(x1, dx1, x2, dx2):
    """Returns the t'-score comparing two measurements.

    Arguments (positional order: x1, dx1, x2, dx2):
      x1  -- the first measurement
      dx1 -- the uncertainty in the first measurement
      x2  -- the second measurement
      dx2 -- the uncertainty in the second measurement

    Can be called positionally or with named arguments in any order:
      t_score(10, 2, 12, 3)                 ->  0.5547
      t_score(x1=10, dx1=2, x2=12, dx2=3)   ->  0.5547
      t_score(x1=10, x2=12, dx1=2, dx2=3)   ->  0.5547
    """
    for name, val in [('x1', x1), ('dx1', dx1), ('x2', x2), ('dx2', dx2)]:
        if not isinstance(val, (int, float, np.number)):
            _fail('t_score',
                f"t_score() needs {name} to be a number.\n"
                f"  got {type(val).__name__}: {_shown(val)}\n"
                "  Argument order: t_score(x1, dx1, x2, dx2)\n"
                "  Example:        t_score(10, 2, 12, 3)  ->  0.5547"
            )
        if _as_finite_number(val) is None:
            _fail('t_score',
                f"t_score() needs {name} to be an ordinary number.\n"
                f"  got {_shown(val)}\n"
                "\n"
                "A nan usually comes from a blank spreadsheet cell, an inf from\n"
                "dividing by zero somewhere earlier, and a value too large to\n"
                "work with from a typing slip. Check that value, then run this\n"
                "cell again."
            )

    if dx1 < 0 or dx2 < 0:
        _w = max(len(str(v)) for v in [x1, dx1, x2, dx2])
        def _row(name, val, highlight=False):
            marker = "   !!! must be >= 0 !!!" if highlight else ""
            return f"  {name:<3} = {str(val):>{_w}}{marker}"
        _warn('t_score',
            "t_score() expects uncertainties to be non-negative.\n"
            + _row('x1',  x1) + "\n"
            + _row('dx1', dx1, highlight=dx1 < 0) + "\n"
            + _row('x2',  x2) + "\n"
            + _row('dx2', dx2, highlight=dx2 < 0)
        )

    if abs(dx1) >= abs(x1) or abs(dx2) >= abs(x2):
        _w = max(len(str(v)) for v in [x1, dx1, x2, dx2])
        def _row(name, val, highlight=False):
            marker = "   !!! unexpectedly large !!!" if highlight else ""
            return f"  {name:<3} = {str(val):>{_w}}{marker}"
        _warn('t_score',
            "t_score() was given uncertainties that look large relative to their\n"
            + "measurements.\n"
            + _row('x1',  x1) + "\n"
            + _row('dx1', dx1, highlight=abs(dx1) >= abs(x1)) + "\n"
            + _row('x2',  x2) + "\n"
            + _row('dx2', dx2, highlight=abs(dx2) >= abs(x2)) + "\n"
            + "\n"
            + "Expected argument order: t_score(x1, dx1, x2, dx2)\n"
            + "If your values are correct, you can ignore this warning."
        )

    if dx1 == 0 and dx2 == 0:
        _fail('t_score',
            "t_score() cannot divide by zero, and dx1 and dx2 are both zero.\n"
            "\n"
            "Check that you have entered your uncertainties correctly."
        )

    with np.errstate(all='ignore'):
        score = abs(x1 - x2) / np.sqrt(dx1**2 + dx2**2)
    return _finite_result(score, 't_score')


def _check_1d_data(data, fn_name, minimum=2):
    """Validate a one-dimensional numeric dataset.

    Returns the data as a float numpy array. Anything unusable shows a
    student-facing error and raises Phys119Error, which stops the cell.

    `minimum` is how many measurements the calling function needs. Anything
    using N-1 needs two; a mean needs one.
    """
    example = f"  Example: {fn_name}([10.1, 10.3, 9.8])"

    if isinstance(data, (int, float, np.number)):
        _fail(fn_name,
            f"{fn_name}() needs a list or numpy array of measurements.\n"
            f"  got the single number {_shown(data)}\n"
            f"{example}"
        )

    try:
        arr = np.asarray(data, dtype=float)
    except OverflowError:
        _fail(fn_name,
            f"{fn_name}() was given a value too large to work with.\n"
            f"  {_shown(data)}\n"
            "\n"
            "Check for a typing slip such as an extra digit or a missing\n"
            "decimal point, then run this cell again."
        )
    except (TypeError, ValueError):
        _fail(fn_name,
            f"{fn_name}() needs a list or numpy array of numbers.\n"
            f"  got {type(data).__name__}: {_shown(data)}\n"
            f"{example}"
        )

    if arr.ndim != 1:
        _fail(fn_name,
            f"{fn_name}() needs one set of measurements, but was given an array\n"
            f"  with shape {arr.shape}.\n"
            "  Pass one set of repeated measurements at a time.\n"
            f"{example}"
        )

    if np.any(np.isnan(arr)):
        _fail(fn_name,
            f"{fn_name}() was given data containing missing values (nan).\n"
            f"  nan values at indices: {np.where(np.isnan(arr))[0].tolist()}\n"
            "\n"
            "Blank cells in a spreadsheet are read in as nan. Fill them in or\n"
            "remove them from the array, then run this cell again."
        )

    if not np.all(np.isfinite(arr)):
        _fail(fn_name,
            f"{fn_name}() was given data containing infinite values.\n"
            f"  infinite values at indices: {np.where(~np.isfinite(arr))[0].tolist()}\n"
            "\n"
            "An inf usually comes from dividing by zero somewhere earlier, or\n"
            "from a spreadsheet cell holding something like 1/0. Check those\n"
            "values, then run this cell again."
        )

    if arr.size < minimum:
        if minimum > 1:
            why = (
                "The spread of a set of measurements cannot be found from fewer\n"
                "than two of them. Check that you passed the full array of\n"
                "repeated measurements."
            )
        else:
            why = (
                "An empty list has no mean. Check that your measurements were\n"
                "read in, then run this cell again."
            )
        measurements = "measurement" if minimum == 1 else "measurements"
        _fail(fn_name,
            f"{fn_name}() needs at least {minimum} {measurements}, got {arr.size}.\n"
            "\n"
            + why
        )

    _warn_if_one_value_stands_out(arr, fn_name)

    return arr


def _warn_if_one_value_stands_out(arr, fn_name):
    """Catch a transcription slip, such as 4336 typed for 433.6.

    Distance is measured from the median, in units of the median absolute
    deviation, so one bad value cannot inflate the scale and hide itself. The
    threshold is deliberately loose: this is a prompt to check a typed number,
    not a suggestion to discard data.
    """
    if arr.size < 4:
        return
    deviations = np.abs(arr - np.median(arr))
    scale = np.median(deviations)
    if scale == 0:
        scale = np.mean(deviations)
    if scale == 0:
        return                       # every value identical, a different warning
    with np.errstate(all='ignore'):
        scores = 0.6745 * deviations / scale
    worst = int(np.argmax(scores))
    if not scores[worst] > 5:
        return
    others = np.delete(arr, worst)
    _warn(fn_name,
        f"{fn_name}() has one measurement far from the others.\n"
        f"  {float(arr[worst]):g} at position {worst}\n"
        f"  the others lie between {float(others.min()):g} and {float(others.max()):g}\n"
        "\n"
        "Check that it was typed correctly. The result below still includes it."
    )


def _warn_if_no_spread(arr, fn_name):
    """Zero spread usually means the instrument cannot resolve the repeats."""
    if arr.size >= 2 and np.all(arr == arr[0]):
        _warn(fn_name,
            f"{fn_name}() got the same value for every measurement, so the\n"
            "spread is exactly zero.\n"
            f"  all {arr.size} measurements = {float(arr[0]):g}\n"
            "\n"
            "That usually means your instrument cannot resolve the difference\n"
            "between repeats, rather than that there is no variation. The\n"
            "result below is still returned."
        )


def _sample_std(arr):
    """Sample standard deviation (N-1 in the denominator) of a 1-D array."""
    N = len(arr)
    return np.sqrt(np.sum((arr - np.mean(arr))**2) / (N - 1))


@_friendly_errors
def mean(data):
    """Returns the mean of a set of measurements.

    Arguments:
      data -- a list or numpy array of measurements (at least 1 value)

    Unlike standard_deviation and standard_unc_of_mean, this one is happy
    with a single measurement, because the mean of one number is that number.
    Everything it refuses, it refuses for the same reasons they do, with the
    same messages.

    Examples:
      mean([10.1, 10.3, 9.8])   ->  10.07
      mean([1, 2, 3, 4])        ->  2.5

    A list, a tuple or a numpy array all work.
    """
    arr = _check_1d_data(data, 'mean', minimum=1)

    with np.errstate(all='ignore'):
        average = np.mean(arr)
    return _finite_result(average, 'mean')


@_friendly_errors
def standard_deviation(data):
    """Returns the standard deviation of a set of repeated measurements.

    Arguments:
      data -- a list or numpy array of measurements (at least 2 values)

    This is the sample standard deviation: the sum of the squared deviations
    from the mean is divided by N - 1, where N is the number of measurements.
    It describes the spread of the individual measurements.

    Examples:
      standard_deviation([10.1, 10.3, 9.8])   ->  0.2517
      standard_deviation([1, 2, 3, 4])        ->  1.291

    A list, a tuple or a numpy array all work.

    Credit: adapted from a function developed by Rebeckah Fussell for
    Cornell Physics Labs.
    """
    arr = _check_1d_data(data, 'standard_deviation')
    _warn_if_no_spread(arr, 'standard_deviation')

    with np.errstate(all='ignore'):
        spread = _sample_std(arr)
    return _finite_result(spread, 'standard_deviation')


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
      standard_unc_of_mean([10.1, 10.3, 9.8])   ->  0.1453
      standard_unc_of_mean([1, 2, 3, 4])        ->  0.6455

    A list, a tuple or a numpy array all work.

    Credit: adapted from a function developed by Rebeckah Fussell for
    Cornell Physics Labs.
    """
    arr = _check_1d_data(data, 'standard_unc_of_mean')
    _warn_if_no_spread(arr, 'standard_unc_of_mean')

    N = len(arr)
    with np.errstate(all='ignore'):
        unc = _sample_std(arr) / np.sqrt(N)
    return _finite_result(unc, 'standard_unc_of_mean')


def _check_xy_data(x, y, dy, fn_name, zero_dy_is_error=False):
    """Validate the x, y and dy a plotting function was given.

    Returns the three as float numpy arrays. Anything unusable shows a
    student-facing error and raises Phys119Error, which stops the cell. The
    one shared check means all three plotting functions accept and refuse
    exactly the same things, with the same messages.

    Zeros in dy are a warning for a plot (the error bar just vanishes) but an
    error for a fit, which divides by dy; `zero_dy_is_error` picks which.
    """
    arrays = []
    for name, value in (('x', x), ('y', y), ('dy', dy)):
        # np.ndim is 0 for a plain number and a 0-d numpy array alike. A
        # ragged list makes it raise; the conversion below reports that.
        try:
            single = np.ndim(value) == 0 and not isinstance(value, str)
        except Exception:
            single = False
        if single:
            if isinstance(value, (int, float, np.number, np.ndarray)):
                got = f"the single number {_shown(value)}"
            else:
                got = f"{type(value).__name__}: {_shown(value)}"
            if name == 'dy':
                _fail(fn_name,
                    f"{fn_name}() needs dy to hold one uncertainty for each data point.\n"
                    f"  got {got}\n"
                    "\n"
                    "Each measurement has its own uncertainty, so dy is a list or\n"
                    "numpy array the same length as y."
                )
            _fail(fn_name,
                f"{fn_name}() needs {name} to be a list or numpy array of values.\n"
                f"  got {got}"
            )

        try:
            arr = np.asarray(value, dtype=float)
        except OverflowError:
            _fail(fn_name,
                f"{fn_name}() was given a value in {name} too large to work with.\n"
                f"  {_shown(value)}\n"
                "\n"
                "Check for a typing slip such as an extra digit or a missing\n"
                "decimal point, then run this cell again."
            )
        except (TypeError, ValueError):
            _fail(fn_name,
                f"{fn_name}() needs {name} to hold numbers only.\n"
                f"  got {type(value).__name__}: {_shown(value)}\n"
                "\n"
                "Check that every entry is a number. Text such as a unit or a\n"
                "note in a spreadsheet column stops it being read as numbers."
            )

        if arr.ndim != 1:
            _fail(fn_name,
                f"{fn_name}() needs {name} to be a single list of values, but was given\n"
                f"  an array with shape {arr.shape}.\n"
                "\n"
                "Pass one column of data at a time."
            )

        if np.any(np.isnan(arr)):
            _fail(fn_name,
                f"{fn_name}() was given missing values (nan) in {name}.\n"
                f"  nan values at indices: {np.where(np.isnan(arr))[0].tolist()}\n"
                "\n"
                "Blank cells in a spreadsheet are read in as nan. Fill them in or\n"
                "remove that data point from x, y and dy, then run this cell again."
            )

        if not np.all(np.isfinite(arr)):
            _fail(fn_name,
                f"{fn_name}() was given infinite values in {name}.\n"
                f"  infinite values at indices: {np.where(~np.isfinite(arr))[0].tolist()}\n"
                "\n"
                "An inf usually comes from dividing by zero somewhere earlier.\n"
                "Check those values, then run this cell again."
            )

        arrays.append(arr)

    x, y, dy = arrays

    if not (len(x) == len(y) == len(dy)):
        _fail(fn_name,
            f"{fn_name}() needs x, y and dy to be the same length.\n"
            f"  len(x)={len(x)}, len(y)={len(y)}, len(dy)={len(dy)}"
        )
    if len(x) == 0:
        _fail(fn_name,
            f"{fn_name}() was given no data: x, y and dy are all empty.\n"
            "\n"
            "Check that your data were read in, then run this cell again."
        )
    if np.any(dy < 0):
        _fail(fn_name,
            f"{fn_name}() was given negative uncertainties in dy.\n"
            f"  Negative values at indices: {np.where(dy < 0)[0].tolist()}\n"
            "  Check that you have passed dy and not some other array."
        )
    if np.any(dy == 0):
        zeros = np.where(dy == 0)[0].tolist()
        if zero_dy_is_error:
            _fail(fn_name,
                f"{fn_name}() was given zeros in dy, and a fit cannot weight data\n"
                "with zero uncertainties.\n"
                f"  Zero values at indices: {zeros}\n"
                "  Check that all uncertainty values are non-zero."
            )
        _warn(fn_name,
            f"{fn_name}() was given zeros in dy, so some error bars\n"
            "will not be visible.\n"
            f"  Zero values at indices: {zeros}\n"
            "  If this is unexpected, check that you have passed dy correctly."
        )

    return x, y, dy


def _check_model_parameters(m, b, fn_name):
    """Make sure the slope m and intercept b are ordinary numbers."""
    for name, value in (('m', m), ('b', b)):
        if not isinstance(value, (int, float, np.number)):
            hint = ""
            if isinstance(value, BestFit):
                hint = (
                    "\n"
                    f"\n{value.name} holds a whole best fit. Use {value.name}.m for its\n"
                    f"slope and {value.name}.b for its intercept."
                )
            elif isinstance(value, str) and _as_finite_number(value) is not None:
                hint = (
                    "\n"
                    f"\n{name} is text here, because of the quotation marks. Write it\n"
                    f"without them: {name}={value.strip()}"
                )
            _fail(fn_name,
                f"{fn_name}() needs {name} to be a number.\n"
                f"  got {type(value).__name__}: {_shown(value)}"
                + hint
            )
        if _as_finite_number(value) is None:
            _fail(fn_name,
                f"{fn_name}() needs {name} to be an ordinary number.\n"
                f"  got {_shown(value)}\n"
                "\n"
                f"Check how {name} was worked out, then run this cell again."
            )


def _check_num_model_points(num_model_points, fn_name):
    """The model line needs a whole number of points, at least two."""
    if (isinstance(num_model_points, bool)
            or not isinstance(num_model_points, (int, np.integer))
            or num_model_points < 2):
        _fail(fn_name,
            f"{fn_name}() needs num_model_points to be a whole number, 2 or more.\n"
            f"  got {_shown(num_model_points)}\n"
            "\n"
            "It is how many points the model line is drawn through. Leaving it\n"
            "out uses 100, which is plenty."
        )


def _student_frame():
    """The frame of the code that called into this module.

    Steps back past every frame belonging to this file, so the decorator and
    one plotting function delegating to another make no difference.
    """
    frame = inspect.currentframe()
    while frame is not None and frame.f_globals is globals():
        frame = frame.f_back
    return frame


def _quiet_parse(source, literal=False):
    """ast.parse (or ast.literal_eval) without warnings. Re-reading the
    student's code must not repeat Python's warnings about it, such as
    "invalid escape sequence" for "$\\Delta$", which they have already had
    (or not) when the cell first ran."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ast.literal_eval(source) if literal else ast.parse(source)


def _typed_call(fn_name, frame):
    """The call to fn_name now running in frame, read from its source.

    Returns the call's ast node, or None when it cannot be found for
    certain: no source to read, or two calls to fn_name on the running line.
    Never raises, because what uses it (labels, a fit's name) is a
    convenience that must not stop the function itself.
    """
    try:
        source = "".join(linecache.getlines(frame.f_code.co_filename))
        tree = _quiet_parse(source)
        line = frame.f_lineno
        calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (getattr(node.func, 'id', None) == fn_name
                 or getattr(node.func, 'attr', None) == fn_name)
            and node.lineno <= line <= node.end_lineno
        ]
        return calls[0] if len(calls) == 1 else None
    except Exception:
        return None


def _typed_names(fn_name, frame):
    """The variable names the student typed for x and y, read from their call.

    Returns {'x': 'DxVec', 'y': 'FVec'} for plot_data(DxVec, FVec, ...),
    with an entry only where a plain variable name was typed. Reading the
    call's own text, rather than searching for a variable that holds the
    array, means the label is always something visible in the student's cell:
    it cannot pick one of IPython's hidden names such as _1, or a different
    alias for the same array. Anything uncertain returns no entry, and the
    caller falls back to a default label.
    """
    call = _typed_call(fn_name, frame)
    if call is None:
        return {}
    typed = dict(zip(('x', 'y'), call.args))
    typed.update((kw.arg, kw.value) for kw in call.keywords if kw.arg in ('x', 'y'))
    return {name: node.id for name, node in typed.items() if isinstance(node, ast.Name)}


def _resolve_labels(fn_name, title, x_label, y_label, data_label, model_label):
    if x_label is None or y_label is None:
        typed = _typed_names(fn_name, _student_frame())
        if x_label is None:
            x_label = typed.get('x', "x (default name)")
        if y_label is None:
            y_label = typed.get('y', "y (default name)")
    if title is None:
        title = "Graph Title"
    if data_label is None:
        data_label = "Data"
    if model_label is None:
        model_label = "y = mx + b (model)"
    # How each label was typed in the cell, so messages can show it as typed
    # and give a fix in the same form (raw or ordinary string).
    typed_labels = _typed_label_pieces(fn_name, _student_frame())
    for name, text in (('title', title), ('x_label', x_label), ('y_label', y_label),
                       ('data_label', data_label), ('model_label', model_label)):
        _check_latex(fn_name, name, text, typed_labels.get(name))
    return title, x_label, y_label, data_label, model_label


# Python reads a single backslash before these letters as a special character
# ("\t" is a tab), so "\theta" in an ordinary string is a tab and "heta".
_ESCAPED_LETTERS = {'\a': 'a', '\b': 'b', '\f': 'f', '\r': 'r', '\t': 't', '\v': 'v', '\n': 'n'}

_MATHS_SPLIT = r'(?<!\\)\$'          # an unescaped $; \$ is a plain dollar sign


def _unescaped_dollars(text):
    """How many $ signs in text start or end maths (a \\$ is a plain $)."""
    return len(re.findall(_MATHS_SPLIT, text))


@functools.lru_cache(maxsize=None)
def _is_latex_symbol(word):
    """True if \\word on its own is something matplotlib's maths can draw,
    such as \\nu or \\Delta (but not \\frac, which needs more)."""
    from matplotlib.mathtext import MathTextParser
    try:
        MathTextParser('path').parse(f'${chr(92)}{word}$')
        return True
    except ValueError:
        return False


def _label_kind(pieces):
    """How a label was typed: 'raw', 'ordinary' or 'mixed' (both), from its
    quoted pieces; None when it was not typed in the call. A piece that is
    only line breaks, "\\n", does not count either way."""
    if not pieces:
        return None
    kinds = set()
    for piece in pieces:
        if 'r' in re.match(r'[A-Za-z]*', piece).group().lower():
            kinds.add('raw')
            continue
        try:
            value = _quiet_parse(piece, literal=True)
        except (ValueError, SyntaxError):
            value = None
        if not (isinstance(value, str) and value and set(value) == {'\n'}):
            kinds.add('ordinary')
    return kinds.pop() if len(kinds) == 1 else ('mixed' if kinds else 'ordinary')


def _as_typed(value, kind):
    """value as a student would type it: raw pieces with "\\n" between them
    for a raw or mixed label, otherwise an ordinary string with each
    backslash doubled."""
    quote = "'" if '"' in value else '"'
    if kind in ('raw', 'mixed'):
        words = []
        for part in re.split(r'(\n+)', value):
            if part.startswith('\n'):
                words.append('"' + '\\n' * len(part) + '"')
            elif part:
                words.append(f'r{quote}{part}{quote}')
        return ' '.join(words)
    return quote + value.replace('\\', '\\\\').replace('\n', '\\n') + quote


def _unescape_latex(text):
    """The label as the student meant it when Python turned a single
    backslash into a special character: a tab back into \\t, and so on.
    A newline is put back as \\n only inside $...$ (\\nu); outside, it is
    a real new line the student asked for."""
    meant, in_maths = [], False
    for index, char in enumerate(text):
        if char == '$' and (index == 0 or text[index - 1] != '\\'):
            in_maths = not in_maths
        if char in _ESCAPED_LETTERS and (char != '\n' or in_maths):
            meant.append('\\' + _ESCAPED_LETTERS[char])
        else:
            meant.append(char)
    return ''.join(meant)


def _plain_commands(text):
    """Each backslash command outside $...$ in text, as (start, end, word):
    LaTeX that matplotlib will show as plain text."""
    found, offset = [], 0
    for index, part in enumerate(re.split(f'({_MATHS_SPLIT})', text)):
        if index % 4 == 0:              # outside maths (parts alternate with $)
            for match in re.finditer(r'\\([A-Za-z]+)', part):
                found.append((offset + match.start(), offset + match.end(), match.group(1)))
        offset += len(part)
    return found


def _label_problem(text, kind):
    """What is wrong with a label, as a dict, or None if it can be drawn as
    meant. 'problem' is a short description; 'fixed' (when known) is the
    label as it was meant, which the message shows in the student's form."""
    from matplotlib import cbook
    from matplotlib.mathtext import MathTextParser
    # A newline only counts when it splits a $...$ pair: "\nu" or "\neq".
    newline_in_maths = (_unescaped_dollars(text) % 2 == 0 and
                        any(_unescaped_dollars(line) % 2 for line in text.split('\n')))
    for char, letter in _ESCAPED_LETTERS.items():
        if char in text and (char != '\n' or newline_in_maths):
            command = re.match(r'[A-Za-z]*', text[text.index(char) + 1:]).group()
            if not command:
                return {'problem': "a line break splits a $...$ pair"}
            return {'problem': f'"\\{letter}{command}" was read as a special character, '
                               f'not the LaTeX command \\{letter}{command}',
                    'escaped': True, 'fixed': _unescape_latex(text)}
    if _unescaped_dollars(text) % 2:
        return {'problem': "a $ has no partner"}

    # Backslashes outside $...$: a \n meant as a new line, or LaTeX that only
    # works between $ signs. \nu is LaTeX; \n(N) or \nForce is a new line.
    for start, end, word in _plain_commands(text):
        if word.startswith('n') and not _is_latex_symbol(word):
            fixed = text
            for s, e, w in reversed(_plain_commands(text)):
                if w.startswith('n') and not _is_latex_symbol(w):
                    fixed = fixed[:s] + '\n' + fixed[s + 2:]
            return {'newline': True, 'fixed': fixed}
        follow = re.match(r' [A-Za-z](?![A-Za-z])', text[end:])
        stop = end + (follow.end() if follow else 0)
        return {'outside': word,
                'fixed': text[:start] + '$' + text[start:stop] + '$' + text[stop:]}

    # Two backslashes before a command inside maths: r"$\\Delta$".
    parts = re.split(f'({_MATHS_SPLIT})', text)
    for index in range(2, len(parts), 4):
        doubled = re.search(r'\\\\([A-Za-z]+)', parts[index])
        if doubled:
            for i in range(2, len(parts), 4):
                parts[i] = re.sub(r'\\\\(?=[A-Za-z])', r'\\', parts[i])
            return {'doubled': doubled.group(1), 'fixed': ''.join(parts)}

    # matplotlib draws each line separately, so check them that way.
    for line in text.split('\n'):
        if not cbook.is_math_text(line):
            continue
        maths = ''.join(re.split(_MATHS_SPLIT, line)[1::2])
        if len(re.findall(r'(?<!\\)\{', maths)) != len(re.findall(r'(?<!\\)\}', maths)):
            return {'problem': "a { has no partner }"}
        try:
            MathTextParser('path').parse(line)
        except ValueError as error:
            detail = str(error).strip().split('\n')[-1]
            detail = re.sub(r'^\w*Exception: ', '', detail)
            detail = re.split(r', found | +\(at char', detail)[0]
            return {'problem': detail if len(detail) <= 60 else "matplotlib could not read it"}
    return None


def _shown_as_typed(text):
    """repr(text), but with the special characters a single backslash makes
    shown the way they were typed: '\\frac' as \\frac, not \\x0crac."""
    shown = _shown(text)
    for code, letter in (('\\x07', 'a'), ('\\x08', 'b'), ('\\x0c', 'f'), ('\\x0b', 'v')):
        shown = shown.replace(code, '\\' + letter)
    return shown


def _check_latex(fn_name, name, text, typed=None):
    """Stop with a short message if a label's LaTeX cannot be drawn, or would
    be drawn as plain text, rather than matplotlib's long error, a blank
    figure, or a label showing \\n or \\Delta.

    typed is (the label's source text, its quoted pieces) when the label was
    typed in the call; the message then shows it as typed and gives the fix
    in the same form, raw or ordinary.
    """
    if not isinstance(text, str):
        return
    found = _label_problem(text, None)
    if found is None:
        return
    kind = _label_kind(typed[1]) if typed else None
    got = f"  got {name}={typed[0]}" if typed else f"  got {_shown_as_typed(text)}"
    fix = (f"  {name}={_as_typed(found['fixed'], kind or 'ordinary')}"
           if found.get('fixed') is not None else None)

    if found.get('newline'):
        explain = {
            'raw': 'In a raw string, \\n is not a new line. Put "\\n" between raw pieces:',
            'ordinary': 'For a new line, write \\n with a single backslash:',
        }.get(kind, 'For a new line, write \\n with a single backslash in an ordinary\n'
                    'string (not a raw one):')
        _fail(fn_name,
            f"{fn_name}() would show \\n as text in {name}, not start a new line.\n"
            f"{got}\n\n{explain}\n{fix}")
    if found.get('outside'):
        _fail(fn_name,
            f"{fn_name}() would show \\{found['outside']} as text in {name}.\n"
            f"{got}\n\nLaTeX commands only work between a pair of $ signs:\n{fix}")
    if found.get('escaped') and kind != 'raw':
        # Correct apart from a missing r: either fix works, so show both.
        _fail(fn_name,
            f"{fn_name}() could not draw the LaTeX in {name}.\n"
            f"{got}\n  problem: {found['problem']}\n\n"
            "Either put r before the opening quotation mark:\n"
            f"  {name}={_as_typed(found['fixed'], 'raw')}\n"
            "or double each backslash:\n"
            f"  {name}={_as_typed(found['fixed'], 'ordinary')}")
    if found.get('doubled'):
        command = found['doubled']
        if kind == 'raw':
            problem = f'\\\\{command} has two backslashes, but this is a raw string (r"...")'
            explain = "In a raw string, each LaTeX command has a single backslash:"
        else:
            problem = f"\\\\{command} reaches matplotlib with two backslashes"
            explain = ("Each LaTeX command needs two backslashes in an ordinary string,\n"
                       'or one in a raw string (r"..."):')
        _fail(fn_name,
            f"{fn_name}() could not draw the LaTeX in {name}.\n"
            f"{got}\n  problem: {problem}\n\n{explain}\n{fix}")

    backslash_rule = {
        'raw': '  - every LaTeX command has a single backslash in a raw string:  r"$\\Delta x$"',
        'mixed': '  - every LaTeX command has a single backslash in r"..." pieces and\n'
                 '    a double backslash in ordinary "..." pieces',
    }.get(kind, '  - every LaTeX command has a double backslash:  "$\\\\Delta x$"')
    _fail(fn_name,
        f"{fn_name}() could not draw the LaTeX in {name}.\n"
        f"{got}\n"
        f"  problem: {found['problem']}\n"
        "\n"
        "Check that:\n"
        '  - every $ has a partner:  "Force $F$ (N)"\n'
        f"{backslash_rule}"
    )


RESIDUAL_LABEL = "residual\n= data - model"


def _reduced_chi2(x, y, dy, m, b, dof):
    """Reduced weighted chi-squared of the line y = m*x + b:
    sum(((y - (m*x + b)) / dy)^2) / dof."""
    with np.errstate(all='ignore'):
        return float(np.sum(((y - (m * x + b)) / dy)**2) / dof)


def _three_sig_figs(value):
    """value to exactly 3 significant figures, trailing zeros kept, as
    (mantissa text, power of ten or None): 1.20 -> ("1.20", None),
    0.0456 -> ("0.0456", None), 1234 -> ("1.23", 3). Powers of ten are used
    from 1000 up and below 0.001."""
    if value == 0:
        return "0.00", None
    rounded = float(f"{value:.3g}")
    exponent = int(np.floor(np.log10(abs(rounded))))
    if exponent >= 3 or exponent < -3:
        return f"{rounded / 10**exponent:.2f}", exponent
    return f"{rounded:.{max(0, 2 - exponent)}f}", None


def _chi2_legend_text(value):
    """The legend line for a reduced chi-squared, in matplotlib's math text."""
    mantissa, exponent = _three_sig_figs(value)
    shown = mantissa if exponent is None else rf"{mantissa} \times 10^{{{exponent}}}"
    return rf"$\chi^2 = {shown}$"


def _parameter_legend_text(name, value):
    """'m = 2.08' for the legend, to exactly 3 significant figures, in
    matplotlib's math text like the chi-squared line. The letter is upright
    (math text's mathrm), to match the m and b in the model's plain-text legend entry."""
    mantissa, exponent = _three_sig_figs(value)
    shown = mantissa if exponent is None else rf"{mantissa} \times 10^{{{exponent}}}"
    return rf"$\mathrm{{{name}}} = {shown}$"


def _value_with_uncertainty(value, uncertainty):
    """value and its uncertainty as text: the uncertainty to 2 significant
    figures, and the value rounded to the same decimal place, as
    ("2.083", "0.031", None). A power of ten shared by both, as for
    _three_sig_figs, is used from 1000 up and below 0.001 (judged by the
    larger of the two): ("2.083", "0.031", 3) for 2083 ± 31."""
    if not (np.isfinite(value) and np.isfinite(uncertainty)) or uncertainty <= 0:
        return f"{value:.4g}", f"{uncertainty:.2g}", None
    exponent = int(np.floor(np.log10(max(abs(value), uncertainty))))
    if -3 <= exponent < 3:
        exponent = None
    else:
        value, uncertainty = value / 10**exponent, uncertainty / 10**exponent
    # The decimal place of the uncertainty's second significant figure;
    # rounding can carry into a new digit (0.0996 -> 0.10), so check again.
    place = int(np.floor(np.log10(uncertainty))) - 1
    if round(uncertainty, -place) >= 10.0**(place + 2):
        place += 1
    decimals = max(0, -place)
    shown_value = round(value, -place) + 0.0          # + 0.0 turns -0.0 into 0.0
    return (f"{shown_value:.{decimals}f}", f"{round(uncertainty, -place):.{decimals}f}",
            exponent)


def _uncertainty_legend_text(name, value, uncertainty):
    """'m = 2.083 ± 0.031' for autofit's legend, in math text like
    _parameter_legend_text."""
    shown_value, shown_uncertainty, exponent = _value_with_uncertainty(value, uncertainty)
    shown = rf"{shown_value} \pm {shown_uncertainty}"
    if exponent is not None:
        shown = rf"({shown}) \times 10^{{{exponent}}}"
    return rf"$\mathrm{{{name}}} = {shown}$"


def _uncertainty_plain_text(name, value, uncertainty):
    """'m = 2.083 ± 0.031' as plain text, for autofit's printed summary."""
    shown_value, shown_uncertainty, exponent = _value_with_uncertainty(value, uncertainty)
    shown = f"{shown_value} ± {shown_uncertainty}"
    if exponent is not None:
        shown = f"({shown})×10^{exponent}"
    return f"{name} = {shown}"


def _draw_data_and_model(ax, x, y, dy, m, b, data_label, model_label,
                         num_model_points, extra_label=None, parameter_labels=None):
    """Draw the data with error bars, the model line if m is given, and the
    legend on ax.

    The legend reads: data, the model, the parameter lines (parameter_labels,
    given whenever m is), then extra_label (the chi-squared). autofit draws
    its own figure, in _autofit_draw, in the same order.
    """
    from matplotlib.lines import Line2D
    text_line = lambda label: Line2D([], [], linestyle="none", label=label)
    data_artist = ax.errorbar(x=x, y=y, yerr=dy, fmt='bo', markersize=3, label=data_label)
    handles = [data_artist]
    if m is not None:
        x_model = np.linspace(start=np.min(x), stop=np.max(x), num=num_model_points)
        with np.errstate(all='ignore'):
            y_model = m * x_model + b
        model_artist, = ax.plot(x_model, y_model, 'r-', label=model_label)
        handles = [data_artist, model_artist] + [text_line(t) for t in parameter_labels or []]
    if extra_label is not None:
        handles.append(text_line(extra_label))
    ax.legend(handles=handles)


def _draw_residuals(ax, x, y, dy, m, b, x_label):
    """The residual panel: data - model, with each point's error bar."""
    with np.errstate(all='ignore'):
        residuals_arr = y - (m * x + b)
    ax.errorbar(x=x, y=residuals_arr, yerr=dy, fmt='bo', markersize=3)
    ax.hlines(y=0, xmin=np.min(x), xmax=np.max(x), color='k')
    ax.set_xlabel(x_label)
    # Two lines, and padded so the label stays clear of the tick numbers.
    ax.set_ylabel(RESIDUAL_LABEL, labelpad=10)
    ax.grid(True, alpha=0.3)


def _plot_data(fn_name, x, y, dy, m, b, title, x_label, y_label, data_label,
               model_label, residuals, num_model_points, figsize, chi2=False):
    """The work behind plot_data and plot_residuals. fn_name is the function
    the student called, so that every message and the axis-label lookup use
    that name."""
    # Everything that can stop the call is checked before the data,
    # whose zero-dy check may show a warning.
    if m is None:
        if chi2:
            _fail(fn_name,
                f"{fn_name}() needs a line to work out chi-squared for, but\n"
                "was given no slope m.\n"
                "\n"
                "Give m (and b if it is not 0), for example\n"
                f"{fn_name}(x, y, dy, m=2, b=0, chi2=True)"
            )
        if b is not None:
            _fail(fn_name,
                f"{fn_name}() was given an intercept b but no slope m.\n"
                f"  got b = {_shown(b)}\n"
                "\n"
                "Give m as well to draw a line, for example\n"
                "plot_data(x, y, dy, m=2, b=0)"
            )
        if residuals:
            _fail(fn_name,
                f"{fn_name}() needs a line to work out residuals from, but\n"
                "was given no slope m.\n"
                "\n"
                "Give m (and b if it is not 0), for example\n"
                + (f"{fn_name}(x, y, dy, m=2, b=0, residuals=True)"
                   if fn_name == 'plot_data' else
                   f"{fn_name}(x, y, dy, m=2, b=0)")
            )
    else:
        # Every line is a 2-parameter model, y = mx + b; leaving b out
        # sets it to 0 but still counts it as a parameter.
        if b is None:
            b = 0
        _check_model_parameters(m, b, fn_name)
    _check_num_model_points(num_model_points, fn_name)
    x, y, dy = _check_xy_data(x, y, dy, fn_name)
    # plot_data's and plot_residuals' own legend wording, which students
    # are shown how to replace with data_label and model_label.
    if data_label is None:
        data_label = "Data"
    if model_label is None:
        model_label = "Model (y=mx+b)"
    title, x_label, y_label, data_label, model_label = _resolve_labels(
        fn_name, title, x_label, y_label, data_label, model_label,
    )
    if chi2 and len(x) <= 2:
        _fail(fn_name,
            f"{fn_name}() needs at least 3 data points to work out chi-squared, got {len(x)}.\n"
            "\n"
            "Reduced chi-squared divides by the number of points minus 2 (the\n"
            "slope and the intercept), which must be at least 1."
        )

    extra = None
    if chi2:
        extra = _chi2_legend_text(_reduced_chi2(x, y, dy, m, b, len(x) - 2))

    # The legend lists the line's parameters, m and b.
    parameters = None
    if m is not None:
        parameters = [_parameter_legend_text("m", m), _parameter_legend_text("b", b)]

    if not residuals:
        fig, ax = plt.subplots(figsize=figsize)
        _draw_data_and_model(ax, x, y, dy, m, b, data_label, model_label,
                             num_model_points, extra, parameters)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)          # the same light grid as with residuals
        plt.show()
        plt.close(fig)
        return

    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, sharex=True,
        figsize=(7, 6) if figsize is None else figsize,
        gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.08},
    )
    _draw_data_and_model(ax_top, x, y, dy, m, b, data_label, model_label,
                         num_model_points, extra, parameters)
    ax_top.set_ylabel(y_label)
    ax_top.set_title(title)
    ax_top.grid(True, alpha=0.3)
    _draw_residuals(ax_bot, x, y, dy, m, b, x_label)
    plt.show()
    plt.close(fig)


@_friendly_errors
def plot_data(
    x, y, dy,
    m=None, b=None,
    title=None,
    x_label=None,
    y_label=None,
    data_label=None,
    model_label=None,
    residuals=False,
    num_model_points=100,
    figsize=None,
    chi2=False,
):
    """Plots data with error bars; adds a straight line y = m*x + b if you
    give a slope m, and the residuals underneath if you ask for them.

    Arguments (positional order: x, y, dy):
      x  -- the x values of your data
      y  -- the y values of your data
      dy -- the uncertainty in each y value, one for every data point

    Optional model line:
      m  -- the slope of a line to draw through the data
      b  -- its y-intercept (default 0 when m is given)
      residuals -- True to add a second graph underneath showing the
                   residuals, data - model (needs m); default False.
                   plot_residuals(x, y, dy, m, b) does the same.

    Each residual is how far a data point lies above (positive) or below
    (negative) the line. Its error bar is the same as that point's dy.

    With a line, the legend lists the data, the model, and the values of m
    and b to 3 significant figures.

    Optional labels, each a piece of text in quotation marks:
      title        -- the graph title (default "Graph Title")
      x_label      -- the x-axis label
      y_label      -- the y-axis label
      data_label   -- the legend entry for the data (default "Data")
      model_label  -- the legend entry for the line
                      (default "Model (y=mx+b)")

    Other options:
      figsize -- the figure size in inches as (width, height); the default is
                 (7, 6) with residuals and the usual size without
      chi2    -- True to add the reduced chi-squared of the line to the
                 legend (needs m); default False. It divides by N - 2, where
                 N is the number of data points.

    Without x_label and y_label, the axes show the variable names you typed
    for x and y, such as DxVec, or "x (default name)" and "y (default name)"
    when what you typed was not a plain variable name.

    Examples:
      plot_data(x=DxVec, y=FVec, dy=dFVec)

      plot_data(x=DxVec, y=FVec, dy=dFVec, m=2, b=0)

      plot_data(
          x=DxVec, y=FVec, dy=dFVec,
          m=2.1, b=0,
          residuals=True,
          title="Hooke's law investigation",
          x_label="Displacement of spring (m)",
          y_label="Force (N)",
      )

    x, y and dy can each be a list or a numpy array.
    """
    _plot_data('plot_data', x, y, dy, m, b, title, x_label, y_label,
               data_label, model_label, residuals, num_model_points, figsize,
               chi2)


# --- How labels were typed, read from the student's call --------------------

_PLOT_DATA_ARGUMENTS = ('x', 'y', 'dy', 'm', 'b', 'title', 'x_label', 'y_label',
                        'data_label', 'model_label')
_LABEL_ARGUMENTS = _PLOT_DATA_ARGUMENTS[5:]
_AUTOFIT_ARGUMENTS = ('x', 'y', 'dy', 'title', 'x_label', 'y_label', 'data_label')


def _typed_label_pieces(fn_name, frame):
    """How each label was typed in the student's call, read from its source:
    {name: (the label's source text, [each quoted piece's source])}.

    Python keeps no trace of the r in r"..." once the code runs, but the
    cell's source still has it. Only labels typed as text in the call are
    listed; one held in a variable cannot be checked this way.
    """
    import io
    import tokenize
    call = _typed_call(fn_name, frame)
    if call is None:
        return {}
    source = "".join(linecache.getlines(frame.f_code.co_filename))
    positional = _AUTOFIT_ARGUMENTS if fn_name == 'autofit' else _PLOT_DATA_ARGUMENTS
    typed = dict(zip(positional, call.args))
    typed.update((kw.arg, kw.value) for kw in call.keywords if kw.arg)
    labels = {}
    for name in _LABEL_ARGUMENTS:
        node = typed.get(name)
        if not (isinstance(node, ast.JoinedStr) or
                (isinstance(node, ast.Constant) and isinstance(node.value, str))):
            continue
        text = ast.get_source_segment(source, node)
        if text is None:
            continue
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        except (tokenize.TokenError, SyntaxError):
            continue
        # FSTRING_START is how Python 3.12 and later begin an f-string.
        starts = {tokenize.STRING, getattr(tokenize, 'FSTRING_START', None)}
        labels[name] = (text, [t.string for t in tokens if t.type in starts])
    return labels


@_friendly_errors
def plot_residuals(
    x, y, dy,
    m, b=None,
    title=None,
    x_label=None,
    y_label=None,
    data_label=None,
    model_label=None,
    num_model_points=100,
    figsize=None,
    chi2=False,
):
    """Plots data with error bars and a straight line y = m*x + b, with the
    residuals in a second graph underneath. The same as plot_data with
    residuals=True.

    Arguments (positional order: x, y, dy, m, b):
      x  -- the x values of your data
      y  -- the y values of your data
      dy -- the uncertainty in each y value, one for every data point
      m  -- the slope of the line
      b  -- the y-intercept of the line (default 0)

    Each residual is data - model: how far a data point lies above (positive)
    or below (negative) the line. Its error bar is the same as that point's
    dy.

    Optional: title, x_label, y_label, data_label, model_label, figsize and
    chi2, all as for plot_data. Without x_label and y_label, the axes show
    the variable names you typed for x and y.

    Examples:
      plot_residuals(x=DxVec, y=FVec, dy=dFVec, m=2, b=0)

    Your final best-fit plot: after fitting the line with manual_fit, use
    plot_residuals with that fit's slope and intercept, a title, axis labels
    with units and a data label:

      plot_residuals(
          x=DxVec, y=FVec, dy=dFVec,
          m=manual_fit1.m, b=manual_fit1.b,
          title="Hooke's law investigation",
          x_label="Displacement of spring (m)",
          y_label="Force (N)",
          data_label="Round 1 data",
      )

    x, y and dy can each be a list or a numpy array.
    """
    _plot_data('plot_residuals', x, y, dy, m, b, title, x_label, y_label,
               data_label, model_label, True, num_model_points, figsize,
               chi2=chi2)


def _weighted_line_fit(x, y, dy):
    """The slope and intercept that minimise chi-squared for a straight line.

    chi^2 = sum(((y - (m*x + b)) / dy)^2) is smallest where both of its
    derivatives are zero, which gives the line exactly, with weights
    w = 1/dy^2. Measured from the weighted mean of x, as here, the slope is

        m = sum(w (x - xbar) (y - ybar)) / sum(w (x - xbar)^2)

    and b follows from the line passing through the weighted means. This
    form avoids the textbook determinant S*Sxx - Sx^2, which loses precision
    when the x values are large compared with their spread (years, say).

    Nothing is squared at full size: only relative weights matter, so they
    are taken as (smallest dy / dy)^2, between 0 and 1, and the distances
    from xbar are measured in units of the largest one. Very large or very
    small values therefore cannot overflow into a wrong answer such as a
    slope of exactly 0. Returns nan values if anything is still not finite.
    """
    with np.errstate(all='ignore'):
        w = (np.min(dy) / dy)**2
        x_bar = np.sum(w * x) / np.sum(w)
        y_bar = np.sum(w * y) / np.sum(w)
        dx = x - x_bar
        scale = np.max(np.abs(dx))
        u = dx / scale
        m = np.sum(w * u * (y - y_bar)) / (scale * np.sum(w * u**2))
        b = y_bar - m * x_bar
    if not (np.isfinite(m) and np.isfinite(b)):
        return float("nan"), float("nan")
    return float(m), float(b)


def _line_fit_uncertainties(x, dy):
    """The uncertainties in the best-fit slope and intercept, dm and db.

    The textbook forms are dm = sqrt(S / Z) and db = sqrt(Sxx / Z), with
    S = sum(1/dy^2), Sxx = sum(x^2/dy^2) and Z = S*Sxx - Sx^2. Measured from
    the weighted mean of x, as in _weighted_line_fit, the same values are

        dm = 1 / sqrt(sum((x - xbar)^2 / dy^2))
        db = sqrt(1 / S + (xbar * dm)^2)

    which avoid Z's loss of precision for x far from zero, and are worked
    out with the same relative weights and scaled distances, so that they
    cannot overflow. Returns nan values if anything is not finite.
    """
    with np.errstate(all='ignore'):
        dy_min = np.min(dy)
        w = (dy_min / dy)**2
        x_bar = np.sum(w * x) / np.sum(w)
        dx = x - x_bar
        scale = np.max(np.abs(dx))
        u = dx / scale
        dm = dy_min / (scale * np.sqrt(np.sum(w * u**2)))
        db = np.sqrt(dy_min**2 / np.sum(w) + (x_bar * dm)**2)
    if not (np.isfinite(dm) and np.isfinite(db)):
        return float("nan"), float("nan")
    return float(dm), float(db)


# Every best fit made by autofit in this kernel, by what identifies it (its
# cell and its call as typed), newest last. The names, and the points left
# out, are also saved beside the notebook with manual_fit's, so a fit keeps
# them after a restart.
_autofits = {}


# ---------------------------------------------------------------------------
# manual_fit: position a line by hand with sliders and number boxes.
#
# The widget is drawn in the student's browser by the JavaScript below, so
# dragging a slider redraws instantly and costs the server nothing; only the
# values of m and b travel to Python. It needs the anywidget package, which
# is imported only when manual_fit is called, so everything else in
# this module works without it.
#
# Each fit is named manual_fit1, manual_fit2, ..., and that name is created
# as a variable in the student's notebook. The call as typed identifies a
# fit, so re-running a cell, even after a kernel restart, gives the same
# name, and the line is restored from a small hidden file next to the
# notebook, as fit_plot did.
# ---------------------------------------------------------------------------

_MANUAL_FIT_ESM = r"""
const BLUE = "#0000ff", RED = "#ff0000", BLACK = "#000000";
const GRID = "rgba(176, 176, 176, 0.3)";
const WIDTH = 560, MAIN_HEIGHT = 330, RESID_HEIGHT = 170;

// The "nice" step (1, 2 or 5 times a power of ten) nearest to giving
// `count` ticks across `span`.
function niceStep(span, count) {
    const raw = span / count;
    const power = Math.pow(10, Math.floor(Math.log10(raw)));
    const n = raw / power;
    return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * power;
}

function decimalsFor(step) {
    return Math.max(0, Math.ceil(-Math.log10(step) - 1e-9));
}

function tickLabel(value, step) {
    const text = value.toFixed(decimalsFor(step));
    return /^-0(\.0*)?$/.test(text) ? text.slice(1) : text;
}

// The smallest 1, 2 or 5 times a power of ten that is at least `value`.
function niceCeil(value) {
    if (!(value > 0)) return 1;
    const power = Math.pow(10, Math.floor(Math.log10(value)));
    for (const n of [1, 2, 5, 10]) {
        if (n * power >= value * (1 - 1e-12)) return n * power;
    }
    return 10 * power;
}

function makeCanvas(height) {
    const canvas = document.createElement("canvas");
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(WIDTH * ratio);
    canvas.height = Math.round(height * ratio);
    canvas.style.width = WIDTH + "px";
    canvas.style.height = height + "px";
    canvas.style.display = "block";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    return [canvas, ctx];
}

// Draw background, grid, ticks, frame and axis labels; return data-to-pixel maps.
function drawAxes(ctx, height, box, xlim, ylim, labels) {
    const { left, right, top, bottom } = box;
    const px = (x) => left + (x - xlim[0]) / (xlim[1] - xlim[0]) * (right - left);
    const py = (y) => bottom - (y - ylim[0]) / (ylim[1] - ylim[0]) * (bottom - top);

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, WIDTH, height);

    ctx.lineWidth = 1;
    ctx.strokeStyle = GRID;
    ctx.fillStyle = BLACK;
    ctx.font = "11px sans-serif";

    const xstep = niceStep(xlim[1] - xlim[0], 6);
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let k = Math.ceil(xlim[0] / xstep); k * xstep <= xlim[1] + 1e-9 * xstep; k++) {
        const x = k * xstep;
        ctx.beginPath(); ctx.moveTo(px(x), top); ctx.lineTo(px(x), bottom); ctx.stroke();
        ctx.fillText(tickLabel(x, xstep), px(x), bottom + 4);
    }
    // About one label per 40 pixels, so the short residual panel gets fewer.
    const ystep = niceStep(ylim[1] - ylim[0], Math.max(2, Math.floor((bottom - top) / 40)));
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let k = Math.ceil(ylim[0] / ystep); k * ystep <= ylim[1] + 1e-9 * ystep; k++) {
        const y = k * ystep;
        ctx.beginPath(); ctx.moveTo(left, py(y)); ctx.lineTo(right, py(y)); ctx.stroke();
        ctx.fillText(tickLabel(y, ystep), left - 5, py(y));
    }

    ctx.strokeStyle = BLACK;
    ctx.strokeRect(left, top, right - left, bottom - top);

    ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    if (labels.x) {
        ctx.textBaseline = "bottom";
        ctx.fillText(labels.x, (left + right) / 2, height - 3);
    }
    if (labels.y) {
        // One or more lines, stacked away from the tick numbers, with the
        // last line nearest the axis and still clear of the numbers.
        const lines = labels.y.split("\n");
        ctx.save();
        ctx.translate(0, (top + bottom) / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textBaseline = "middle";
        lines.forEach((line, i) => {
            ctx.fillText(line, 0, 12 + 14 * i);
        });
        ctx.restore();
    }
    return [px, py];
}

function clipTo(ctx, box) {
    ctx.beginPath();
    ctx.rect(box.left, box.top, box.right - box.left, box.bottom - box.top);
    ctx.clip();
}

function drawPoints(ctx, box, px, py, xs, ys, errs) {
    ctx.save();
    clipTo(ctx, box);
    ctx.strokeStyle = BLUE; ctx.fillStyle = BLUE; ctx.lineWidth = 1.5;
    for (let i = 0; i < xs.length; i++) {
        const X = px(xs[i]);
        ctx.beginPath();
        ctx.moveTo(X, py(ys[i] - errs[i]));
        ctx.lineTo(X, py(ys[i] + errs[i]));
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(X, py(ys[i]), 2.5, 0, 2 * Math.PI);
        ctx.fill();
    }
    ctx.restore();
}

// A value to exactly 3 significant figures, trailing zeros kept, with a
// power of ten from 1000 up and below 0.001 (as _three_sig_figs in Python).
function threeSigFigs(value) {
    if (value === 0) return "0.00";
    const rounded = Number(value.toPrecision(3));
    const exponent = Math.floor(Math.log10(Math.abs(rounded)));
    if (exponent >= 3 || exponent < -3) {
        const sup = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³",
                      "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸",
                      "9": "⁹" };
        const power = String(exponent).split("").map((c) => sup[c]).join("");
        return (rounded / Math.pow(10, exponent)).toFixed(2) + "×10" + power;
    }
    return rounded.toFixed(Math.max(0, 2 - exponent));
}

// The legend, as plot_data's reads: the data, the model line, then lines of
// text (m, b and, if asked for, chi-squared).
function drawLegend(ctx, box, dataLabel, modelLabel, textLines) {
    ctx.font = "11px sans-serif";
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    const lx = box.left + 8, ly = box.top + 12;
    const texts = [dataLabel, modelLabel].concat(textLines);
    const width = 24 + Math.max(...texts.map((t) => ctx.measureText(t).width));
    const height = 18 * texts.length;
    ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
    ctx.strokeStyle = "#cccccc"; ctx.lineWidth = 1;
    ctx.fillRect(lx - 4, ly - 9, width + 8, height);
    ctx.strokeRect(lx - 4, ly - 9, width + 8, height);
    ctx.strokeStyle = BLUE; ctx.fillStyle = BLUE; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(lx + 8, ly - 5); ctx.lineTo(lx + 8, ly + 5); ctx.stroke();
    ctx.beginPath(); ctx.arc(lx + 8, ly, 2.5, 0, 2 * Math.PI); ctx.fill();
    ctx.strokeStyle = RED;
    ctx.beginPath(); ctx.moveTo(lx, ly + 18); ctx.lineTo(lx + 16, ly + 18); ctx.stroke();
    ctx.fillStyle = BLACK;
    texts.forEach((text, i) => ctx.fillText(text, lx + 24, ly + 18 * i));
}

function render({ model, el }) {
    const xs = model.get("x"), ys = model.get("y"), errs = model.get("dy");
    const labels = model.get("labels");

    // A fixed view of the data, padded 5% as matplotlib does, so the axes
    // do not jump about while the line moves.
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const xpad = 0.05 * ((xmax - xmin) || 1);
    const xlim = [xmin - xpad, xmax + xpad];
    const lo = Math.min(...ys.map((y, i) => y - errs[i]));
    const hi = Math.max(...ys.map((y, i) => y + errs[i]));
    const ypad = 0.05 * ((hi - lo) || 1);
    const ylim = [lo - ypad, hi + ypad];

    // The left margin leaves room for a two-line axis label beside the numbers.
    const mainBox = { left: 76, right: WIDTH - 14, top: 10, bottom: MAIN_HEIGHT - 26 };
    const residBox = { left: 76, right: WIDTH - 14, top: 10, bottom: RESID_HEIGHT - 40 };
    const [mainCanvas, mainCtx] = makeCanvas(MAIN_HEIGHT);
    const [residCanvas, residCtx] = makeCanvas(RESID_HEIGHT);

    // The line naming this fit is printed by Python as ordinary output, not
    // drawn here, so that it survives saving and exporting the notebook.

    // One slider-and-box row per parameter.
    function controlRow(param, labelText) {
        const range = model.get(param + "_range");     // [min, max, step]
        const decimals = decimalsFor(range[2]);
        const row = document.createElement("div");
        row.style.display = "flex";
        row.style.alignItems = "center";
        row.style.gap = "10px";
        row.style.width = WIDTH + "px";
        row.style.margin = "4px 0";
        const label = document.createElement("span");
        label.textContent = labelText;
        label.style.width = "7.5em";
        label.style.fontFamily = "sans-serif";
        label.style.fontSize = "13px";
        const slider = document.createElement("input");
        slider.type = "range";
        // The slider counts positions 0, 1, ..., N, and each value is worked
        // out as min + k * step. The last position then lands exactly on the
        // maximum whatever the step, which a slider stepping in a decimal
        // such as 1/11 cannot do.
        const [lo, hi, step] = range;
        const positions = Math.round((hi - lo) / step);
        const valueAt = (k) => parseFloat((lo + k * step).toPrecision(12));
        slider.min = 0; slider.max = positions; slider.step = 1;
        slider.style.flex = "1";
        slider.setAttribute("aria-label", labelText);
        const box = document.createElement("input");
        box.type = "number";
        box.step = "any";
        box.style.width = "8em";
        box.setAttribute("aria-label", labelText + " value");
        // Keys typed in the box must not reach JupyterLab's notebook
        // shortcuts, which include the number keys.
        for (const kind of ["keydown", "keyup", "keypress"]) {
            box.addEventListener(kind, (event) => event.stopPropagation());
        }
        const send = (value) => {
            if (!Number.isFinite(value)) return;
            model.set(param, value);
            model.save_changes();
        };
        // The box's arrows move by one slider step and keep any typed
        // offset: from a typed 9.937 with a step of 0.05, up gives 9.987.
        box.step = range[2];
        let last = model.get(param);       // kept up to date by show()
        slider.addEventListener("input", () => {
            const k = parseInt(slider.value, 10);
            send(k === positions ? hi : k === 0 ? lo : valueAt(k));
        });
        box.addEventListener("change", () => send(parseFloat(box.value)));
        // An arrow key or a click on the box's spinner fires "input" with no
        // inputType (typing has one). The browser's own step would snap to a
        // grid, so the new value is worked out here from the last one.
        box.addEventListener("input", (event) => {
            if (event.inputType) return;
            const value = parseFloat(box.value);
            if (!Number.isFinite(value) || value === last) return;
            const next = parseFloat((last + (value > last ? 1 : -1) * range[2]).toFixed(12));
            box.value = String(next);
            send(next);
        });
        const show = () => {
            const value = model.get(param);
            last = value;
            // A typed value beyond the slider's range widens the range, so
            // nothing typed is refused or rounded.
            const k = (value - lo) / step;
            if (k < parseFloat(slider.min)) slider.min = Math.floor(k);
            if (k > parseFloat(slider.max)) slider.max = Math.ceil(k);
            slider.value = Math.round(k);
            if (document.activeElement !== box) {
                const rounded = parseFloat(value.toFixed(decimals));
                box.value = rounded === value ? value.toFixed(decimals) : String(value);
            }
        };
        row.append(label, slider, box);
        return [row, show];
    }

    const [slopeRow, showSlope] = controlRow("m", "slope m");
    const [interceptRow, showIntercept] = controlRow("b", "intercept b");

    // Reset, under the residuals so it costs no space between the plots:
    // back to the middle of the ranges.
    const resetRow = document.createElement("div");
    resetRow.style.width = WIDTH + "px";
    resetRow.style.textAlign = "right";
    resetRow.style.margin = "4px 0 0 0";
    const reset = document.createElement("button");
    reset.type = "button";
    reset.textContent = "Reset slope and intercept";
    reset.title = "Put the line back in the middle of the slider ranges";
    reset.addEventListener("click", () => {
        const start = model.get("start");
        model.set("m", start[0]);
        model.set("b", start[1]);
        model.save_changes();
    });
    resetRow.append(reset);

    function redraw() {
        const m = model.get("m"), b = model.get("b");

        // Main plot: the model line across the data's x range, then the data.
        const [px, py] = drawAxes(mainCtx, MAIN_HEIGHT, mainBox, xlim, ylim,
                                  { y: labels.y_label });
        mainCtx.save();
        clipTo(mainCtx, mainBox);
        mainCtx.strokeStyle = RED; mainCtx.lineWidth = 1.5;
        mainCtx.beginPath();
        mainCtx.moveTo(px(xmin), py(m * xmin + b));
        mainCtx.lineTo(px(xmax), py(m * xmax + b));
        mainCtx.stroke();
        mainCtx.restore();
        drawPoints(mainCtx, mainBox, px, py, xs, ys, errs);
        // Reduced chi-squared of the line where it is now, if asked for.
        let chi2Label = null;
        const dof = model.get("dof");
        if (labels.chi2 && dof >= 1) {
            let sum = 0;
            for (let i = 0; i < xs.length; i++) {
                const r = (ys[i] - (m * xs[i] + b)) / errs[i];
                sum += r * r;
            }
            chi2Label = "χ² = " + threeSigFigs(sum / dof);
        }
        const textLines = ["m = " + threeSigFigs(m), "b = " + threeSigFigs(b)];
        drawLegend(mainCtx, mainBox, labels.data_label, labels.model_label,
                   chi2Label ? textLines.concat([chi2Label]) : textLines);

        // Residuals, data - model, on a symmetric scale in 1-2-5 steps so it
        // changes in jumps rather than wobbling with every slider movement.
        const resid = ys.map((y, i) => y - (m * xs[i] + b));
        const reach = Math.max(...resid.map((r, i) => Math.abs(r) + errs[i]));
        const half = niceCeil(1.1 * reach);
        const [rx, ry] = drawAxes(residCtx, RESID_HEIGHT, residBox, xlim, [-half, half],
                                  { x: labels.x_label, y: labels.residual_y_label });
        residCtx.strokeStyle = BLACK; residCtx.lineWidth = 1.5;
        residCtx.beginPath();
        residCtx.moveTo(rx(xmin), ry(0)); residCtx.lineTo(rx(xmax), ry(0));
        residCtx.stroke();
        drawPoints(residCtx, residBox, rx, ry, xs, resid, errs);

        showSlope();
        showIntercept();
    }

    const onChange = () => redraw();
    model.on("change:m", onChange);
    model.on("change:b", onChange);

    const wrapper = document.createElement("div");
    wrapper.style.width = WIDTH + "px";
    wrapper.append(mainCanvas, slopeRow, interceptRow, residCanvas, resetRow);
    el.append(wrapper);
    redraw();

    return () => {
        model.off("change:m", onChange);
        model.off("change:b", onChange);
    };
}

export default { render };
"""

_manual_fit_widget_class = None


def _manual_fit_widget():
    """The widget class, built on first use so anywidget is only needed then."""
    global _manual_fit_widget_class
    if _manual_fit_widget_class is None:
        import anywidget
        import traitlets

        class ManualFitWidget(anywidget.AnyWidget):
            _esm = _MANUAL_FIT_ESM
            name = traitlets.Unicode("").tag(sync=True)
            m = traitlets.Float(0.0).tag(sync=True)
            b = traitlets.Float(0.0).tag(sync=True)
            start = traitlets.List([0.0, 0.0]).tag(sync=True)
            x = traitlets.List([]).tag(sync=True)
            y = traitlets.List([]).tag(sync=True)
            dy = traitlets.List([]).tag(sync=True)
            m_range = traitlets.List([-1.0, 1.0, 0.01]).tag(sync=True)
            b_range = traitlets.List([-1.0, 1.0, 0.01]).tag(sync=True)
            labels = traitlets.Dict({}).tag(sync=True)
            dof = traitlets.Int(1).tag(sync=True)

        _manual_fit_widget_class = ManualFitWidget
    return _manual_fit_widget_class


def _nice_step(span, count):
    """A 1, 2 or 5 times a power of ten step giving about `count` steps."""
    raw = span / count
    power = 10 ** np.floor(np.log10(raw))
    n = raw / power
    return float((5 if n >= 5 else 2 if n >= 2 else 1) * power)


def _tidy(value):
    """A float without binary noise such as 3.3850000000000002."""
    return float(f"{value:.12g}")


def _manual_fit_ranges(x, y, dy):
    """Automatic slider ranges for m and b, from the data alone.

    The slope range is symmetric about zero, three times the steepest slope
    the data's spread could suggest, so its centre gives nothing away. The
    intercept range covers the data's vertical extent plus whatever that
    slope range can move the line by at the furthest x.
    """
    x_span = float(np.max(x) - np.min(x))
    lo = float(np.min(y - dy))
    hi = float(np.max(y + dy))
    y_span = (hi - lo) or 1.0
    slope_limit = 3 * y_span / x_span
    m_step = _nice_step(2 * slope_limit, 1000)
    m_limit = float(np.ceil(slope_limit / m_step) * m_step)

    reach = m_limit * float(np.max(np.abs(x)))
    b_lo, b_hi = lo - reach, hi + reach
    b_step = _nice_step(b_hi - b_lo, 1000)
    return ([_tidy(-m_limit), _tidy(m_limit), m_step],
            [_tidy(np.floor(b_lo / b_step) * b_step),
             _tidy(np.ceil(b_hi / b_step) * b_step), b_step])


def _nice_step_up(value):
    """The smallest 1, 2 or 5 times a power of ten that is at least value."""
    power = 10 ** np.floor(np.log10(value))
    for n in (1, 2, 5, 10):
        if n * power >= value * (1 - 1e-12):
            return float(n * power)


def _slider_step(span, effect_per_unit, smallest_error):
    """A slider step that one move of it is just visible in the residuals.

    One step changes the furthest residual by about a tenth of the smallest
    error bar, rounded down to a 1, 2 or 5 step, then kept between about 50
    and 500 positions across the range so that a very wide range stays
    usable and a very narrow one stays smooth.
    """
    step = _nice_step(0.1 * smallest_error / effect_per_unit, 1)
    if span / step > 500:
        step = _nice_step_up(span / 500)
    if span / step < 50:
        step = _nice_step(span / 50, 1)
    return float(step)


def _manual_fit_slider_ranges(x, y, dy, m_min, m_max, b_min, b_max,
                              m_steps=None, b_steps=None):
    """Slider ranges, from the keywords given and the data for the rest.

    Returns ([m_min, m_max, m_step], [b_min, b_max, b_step]). Each step comes
    from the data (see _slider_step), unless m_steps or b_steps fixes the
    number of positions on that slider.
    """
    (auto_m_min, auto_m_max, _), (auto_b_min, auto_b_max, _) = \
        _manual_fit_ranges(x, y, dy)
    smallest_error = float(np.min(dy))
    furthest_x = float(np.max(np.abs(x))) or 1.0     # the line turns about x = 0
    ranges = []
    for name, lo, hi, auto_lo, auto_hi, steps, effect in (
        ('m', m_min, m_max, auto_m_min, auto_m_max, m_steps, furthest_x),
        ('b', b_min, b_max, auto_b_min, auto_b_max, b_steps, 1.0),
    ):
        if steps is not None and (isinstance(steps, bool)
                                  or not isinstance(steps, (int, np.integer))
                                  or steps < 2):
            _fail('manual_fit',
                f"manual_fit() needs {name}_steps to be a whole number, 2 or more.\n"
                f"  got {_shown(steps)}\n"
                "\n"
                f"It is how many positions the {name} slider has. Leaving it out\n"
                "chooses them from your data."
            )
        for key, value in ((f'{name}_min', lo), (f'{name}_max', hi)):
            if value is None:
                continue
            if (isinstance(value, bool) or not isinstance(value, (int, float, np.number))
                    or _as_finite_number(value) is None):
                _fail('manual_fit',
                    f"manual_fit() needs {key} to be an ordinary number.\n"
                    f"  got {type(value).__name__}: {_shown(value)}"
                )
        final_lo = auto_lo if lo is None else float(lo)
        final_hi = auto_hi if hi is None else float(hi)
        if not final_lo < final_hi:
            where = lambda given, value: "" if given is not None else " (chosen from the data)"
            _fail('manual_fit',
                f"manual_fit() needs {name}_min to be below {name}_max.\n"
                f"  {name}_min = {final_lo:g}{where(lo, final_lo)}\n"
                f"  {name}_max = {final_hi:g}{where(hi, final_hi)}\n"
                "\n"
                f"Check the values given for {name}_min and {name}_max."
            )
        span = final_hi - final_lo
        # steps counts positions, both ends included: 1.5 to 2.5 with 11
        # positions is a step of 0.1.
        step = (span / (int(steps) - 1) if steps is not None
                else _slider_step(span, effect, smallest_error))
        ranges.append([_tidy(final_lo), _tidy(final_hi), _tidy(step)])
    return ranges[0], ranges[1]


# Fits created in this kernel: key (the call as typed) -> ManualFit.
_manual_fits = {}


def _manual_fit_store_path(frame):
    """The hidden file that remembers this notebook's fits, or None.

    Named after the notebook, in the folder the kernel runs in (the
    notebook's own folder in Jupyter), as fit_plot's cache files are. None
    when the notebook cannot be identified, as outside Jupyter; the fits are
    then remembered for this kernel only.
    """
    try:
        notebook = os.environ.get("JPY_SESSION_NAME")
        if not notebook and frame is not None:
            notebook = frame.f_globals.get("__vsc_ipynb_file__")   # VS Code
        if not notebook:
            return None
        stem = os.path.splitext(os.path.basename(notebook))[0]
        return os.path.join(os.getcwd(), f".{stem}.manual_fits.json")
    except Exception:
        return None


def _manual_fit_store_read(path):
    """Saved fits as {key: {"name": ..., "m": ..., "b": ...}}; {} if none."""
    if path is None:
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            stored = json.load(f)
        fits = stored.get("fits", {})
        return fits if isinstance(fits, dict) else {}
    except Exception:
        return {}


def _manual_fit_store_write(path, fits):
    """Save fits. A failure (say a read-only folder) only loses the memory."""
    if path is None:
        return
    try:
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "fits": fits}, f, indent=1)
        os.replace(temporary, path)
    except Exception:
        pass


def _current_cell_id():
    """The id of the notebook cell now running, or None.

    JupyterLab and Notebook 7 send each cell's permanent id (saved in the
    notebook file) with every request to run it. Elsewhere there is none.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()
        kernel = getattr(shell, "kernel", None)
        parent = kernel.get_parent() if kernel is not None else None
        if not parent:
            parent = getattr(shell, "parent_header", None)
        cell = (parent or {}).get("metadata", {}).get("cellId")
        return cell if isinstance(cell, str) and cell else None
    except Exception:
        return None


def _manual_fit_key(frame, x, y, dy, m_range, b_range):
    """What identifies a fit: its notebook cell and its call as typed.

    The cell keeps identical calls in different cells apart (a best, a
    lowest and a highest slope fit of the same data, say), and re-running a
    cell, even after a restart, finds the same fit. The call is read from
    its text, ignoring layout. Without a cell id (outside JupyterLab) the
    call alone identifies the fit; when the call's text cannot be read
    either, the data and ranges stand in for it.
    """
    cell = _current_cell_id()
    where = f"cell {cell} | " if cell else ""
    call = _typed_call('manual_fit', frame)
    if call is not None:
        try:
            return where + "call: " + ast.unparse(call)
        except Exception:
            pass
    return where + "data: " + json.dumps(
        [x.tolist(), y.tolist(), dy.tolist(), m_range, b_range])


def _manual_fit_name(key, stored, names_in_use, prefix="manual_fit"):
    """The fit's name: its saved one, or the next free manual_fitN (or,
    for autofit, autofitN)."""
    entry = stored.get(key)
    if isinstance(entry, dict) and isinstance(entry.get("name"), str):
        return entry["name"]
    number = 1
    while f"{prefix}{number}" in names_in_use:
        number += 1
    return f"{prefix}{number}"


class ManualFit:
    """A line you positioned by hand with manual_fit.

    .m and .b give its slope and intercept, wherever you left the sliders,
    and .chi2 the reduced chi-squared of the line there:

      manual_fit1.m      the slope
      manual_fit1.b      the intercept
      manual_fit1.chi2   its reduced chi-squared

    Run help(manual_fit) for how the sliders, the name and the
    memory of your line work.
    """

    # No per-object dictionary, which also keeps help() free of the
    # __dict__ and __weakref__ entries that would mean nothing to a student.
    __slots__ = ("name", "_widget")

    def __init__(self, name, widget):
        self.name = name
        self._widget = widget

    @property
    def m(self):
        """The slope, wherever you left the slope slider."""
        return float(self._widget.m)

    @property
    def b(self):
        """The intercept, wherever you left the intercept slider."""
        return float(self._widget.b)

    @property
    def chi2(self):
        """The reduced chi-squared of the line where you left it (divided by
        N - 2), or None with fewer than 3 data points."""
        w = self._widget
        if w.dof < 1:
            return None
        return _reduced_chi2(np.array(w.x), np.array(w.y), np.array(w.dy),
                             w.m, w.b, w.dof)

    def __repr__(self):
        return f"{self.name}: m = {self.m:.4g}, b = {self.b:.4g}"


def _manual_fit_picture(x, y, dy, m, b, labels):
    """A still picture of the fit, as PNG bytes, for exports that cannot
    show the widget (no saved widget state, viewed offline, or PDF).

    Drawn on a bare matplotlib Figure with the Agg canvas, never through
    pyplot, so no figure is registered, shown, left open or switched to a
    different display mode. It mirrors plot_data's figure with residuals.
    """
    import io
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(7, 6))
    FigureCanvasAgg(fig)
    ax_top, ax_bot = fig.subplots(
        nrows=2, ncols=1, sharex=True,
        gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.08},
    )
    extra = None
    if labels.get("chi2") and len(x) > 2:
        extra = _chi2_legend_text(_reduced_chi2(x, y, dy, m, b, len(x) - 2))
    _draw_data_and_model(ax_top, x, y, dy, m, b, labels["data_label"],
                         labels["model_label"], 100, extra,
                         [_parameter_legend_text("m", m), _parameter_legend_text("b", b)])
    ax_top.set_ylabel(labels["y_label"])
    ax_top.grid(True, alpha=0.3)
    _draw_residuals(ax_bot, x, y, dy, m, b, labels["x_label"])
    buffer = io.BytesIO()
    # Cropped to what is drawn, so the two-line residual label is never cut.
    fig.savefig(buffer, format="png", dpi=80, bbox_inches="tight")
    return buffer.getvalue()


def _announce_manual_fit(name, m, b):
    """Say which variable holds the fit, and where its line starts.

    Printed as ordinary output rather than drawn by the widget, so it is
    kept when the notebook is saved or exported, where the widget itself is
    not. After a re-run the starting values are where the line was left.
    """
    final = (f"{name}.m and {name}.b hold your final slope and intercept "
             "values, wherever you leave the sliders.")
    started = f"Line started at: m = {m:g}, b = {b:g}"
    pair = _notebook_display()
    if pair is None:
        print(final)
        print(started)
        return
    display, HTML = pair
    code = lambda text: f"<code>{html.escape(text)}</code>"
    # The starting values mostly matter in an export, so they are quieter.
    display(HTML(
        '<div style="font-family: sans-serif; font-size: 13px;">'
        f"{code(name + '.m')} and {code(name + '.b')} hold your final slope and "
        "intercept values, wherever you leave the sliders.</div>"
        '<div style="font-family: sans-serif; font-size: 12px; '
        'color: var(--jp-ui-font-color2, #757575);">'
        f"Line started at: m = {m:g}, b = {b:g}</div>"
    ))


@_friendly_errors
def manual_fit(x, y, dy, m_min=None, m_max=None, b_min=None, b_max=None,
               m_steps=None, b_steps=None, chi2=False):
    """Lets you position a straight line y = m*x + b on your data by hand,
    using sliders, with the residuals shown underneath.

    Arguments (positional order: x, y, dy):
      x  -- the x values of your data
      y  -- the y values of your data
      dy -- the uncertainty in each y value, one for every data point

    Optional slider ranges, each a number:
      m_min, m_max -- the lowest and highest slope on the slope slider
      b_min, b_max -- the lowest and highest intercept on the intercept slider
    Any you leave out are chosen from your data. The line starts in the
    middle of both ranges.

    Each step of a slider moves the residuals by about a tenth of the
    smallest error bar, so every step is a small but real change; to set
    the number of positions on a slider yourself, give:
      m_steps, b_steps -- the number of positions on that slider, counting
                          both ends: m_min=1.5, m_max=2.5, m_steps=11 moves
                          in steps of 0.1

    Beside each slider is a box: type an exact value there and press Enter.
    Its up and down arrows move the value by one slider step, keeping any
    extra digits you typed. A value outside the slider's range is accepted,
    and the slider stretches to include it.

    Optional:
      chi2 -- True to show the reduced chi-squared of the line in the
              legend, updating as you move it (divided by N - 2, where N is
              the number of data points); default False

    You do not need to store anything. Each fit is given a name, shown above
    its graph: manual_fit1, manual_fit2, and so on. In later cells, that
    name gives the line's slope and intercept, wherever you left them:

      manual_fit1.m      the slope
      manual_fit1.b      the intercept
      manual_fit1.chi2   its reduced chi-squared

    Use them for your final best-fit plot, for example
    plot_data(DxVec, FVec, dFVec, m=manual_fit1.m, b=manual_fit1.b,
              residuals=True)
    with a title, axis labels and a data label (see help(plot_data)).

    Your line is remembered: running the cell again, even after restarting
    the kernel, brings it back where you left it, under the same name. The
    "Reset slope and intercept" button puts it back in the middle of the
    ranges.

    Examples:
      manual_fit(x=DxVec, y=FVec, dy=dFVec)

      manual_fit(DxVec, FVec, dFVec, m_min=1.5, m_max=2.5,
                        b_min=-0.2, b_max=0.2)

      fitting_boundaries = dict(m_min=1.5, m_max=2.5, b_min=-0.2, b_max=0.2)
      manual_fit(DxVec, FVec, dFVec, **fitting_boundaries)

      manual_fit(DxVec, FVec, dFVec, chi2=True)

    The axes are labelled with the variable names you typed for x and y.
    x, y and dy can each be a list or a numpy array.
    """
    x, y, dy = _check_xy_data(x, y, dy, 'manual_fit')
    if np.all(x == x[0]):
        _fail('manual_fit',
            "manual_fit() needs at least two different x values.\n"
            "\n"
            "A slope cannot be judged when every point has the same x. Check\n"
            "that you passed the right array as x."
        )
    m_range, b_range = _manual_fit_slider_ranges(x, y, dy, m_min, m_max, b_min, b_max,
                                                 m_steps, b_steps)
    if chi2 and len(x) < 3:
        _fail('manual_fit',
            f"manual_fit() needs at least 3 data points to work out chi-squared, got {len(x)}.\n"
            "\n"
            "Reduced chi-squared divides by the number of points minus 2 (the\n"
            "slope and the intercept), which must be at least 1."
        )

    try:
        widget_class = _manual_fit_widget()
    except ImportError:
        _fail('manual_fit',
            "manual_fit() needs the anywidget package, which is not\n"
            "installed in this Python environment.\n"
            "\n"
            "This is not a mistake in your code. Let your instructor or TA know."
        )

    frame = _student_frame()
    typed = _typed_names('manual_fit', frame)
    key = _manual_fit_key(frame, x, y, dy, m_range, b_range)
    path = _manual_fit_store_path(frame)
    stored = _manual_fit_store_read(path)

    names_in_use = {e.get("name") for e in stored.values() if isinstance(e, dict)}
    names_in_use |= {fit.name for k, fit in _manual_fits.items() if k != key}
    if key in _manual_fits:
        name = _manual_fits[key].name
    else:
        name = _manual_fit_name(key, stored, names_in_use)

    # Start in the middle of the ranges, unless this fit has been moved
    # before, in which case put the line back where it was left.
    start = [_tidy((m_range[0] + m_range[1]) / 2), _tidy((b_range[0] + b_range[1]) / 2)]
    m0, b0 = start
    entry = stored.get(key)
    if isinstance(entry, dict):
        saved_m = _as_finite_number(entry.get("m"))
        saved_b = _as_finite_number(entry.get("b"))
        if saved_m is not None and saved_b is not None:
            m0, b0 = saved_m, saved_b

    widget = widget_class(
        name=name, m=m0, b=b0, start=start,
        x=x.tolist(), y=y.tolist(), dy=dy.tolist(),
        m_range=m_range, b_range=b_range,
        dof=len(x) - 2,
        labels={
            "x_label": typed.get('x', "x (default name)"),
            "y_label": typed.get('y', "y (default name)"),
            "data_label": "Data", "model_label": "Model (y=mx+b)",
            "residual_y_label": RESIDUAL_LABEL,
            "chi2": bool(chi2),
        },
    )
    fit = ManualFit(name, widget)
    _manual_fits[key] = fit

    def remember(change=None):
        fits = _manual_fit_store_read(path)
        fits[key] = {"name": name, "m": float(widget.m), "b": float(widget.b)}
        _manual_fit_store_write(path, fits)

    widget.observe(remember, names=["m", "b"])
    remember()

    # The fit's name becomes a variable in the student's notebook.
    if frame is not None:
        frame.f_globals[name] = fit

    _announce_manual_fit(name, m0, b0)
    pair = _notebook_display()
    if pair is not None:
        display, _ = pair
        # One output holding the widget, a still picture of the line as it
        # starts, and a sentence. A running notebook shows the widget; an
        # export with saved widget state shows the widget too; anything that
        # cannot show a widget falls back to the picture.
        bundle = widget._repr_mimebundle_()
        bundle = bundle[0] if isinstance(bundle, tuple) else bundle
        try:
            bundle["image/png"] = _manual_fit_picture(x, y, dy, m0, b0, widget.labels)
        except Exception:
            pass                      # the widget still works without it
        bundle["text/plain"] = (f"{name}: a line fitted by hand with manual_fit "
                                f"(the plot appears in a running notebook)")
        display(bundle, raw=True)


# ---------------------------------------------------------------------------
# autofit: the best-fit line, with an interface for leaving points out of
# the fit and putting them back.
#
# The figure is drawn by matplotlib in Python, as plot_data draws it (except
# that, once points are left out, the residuals graph is scaled to the
# points in the fit), and sent to the browser as a picture; the widget's JavaScript only shows
# it, finds which point was clicked, and lists the points with checkboxes.
# A change of selection goes to Python, which refits, redraws and sends the
# new picture back. Clicks are occasional, unlike a slider's drag, so the
# round trip costs little, and in return LaTeX in labels works as in
# plot_data, and the still picture kept for exports is the same figure.
#
# Results are named autofit1, autofit2, ...
# The points left out are remembered beside the notebook, with the fit's
# name, so running the cell again (even after a restart) leaves them out
# again.
# ---------------------------------------------------------------------------

_AUTOFIT_ESM = r"""
const GREY = "#888888";

function render({ model, el }) {
    const n = model.get("n");
    const points = model.get("points");           // [[x, y, dy] as text, ...]
    const xValues = model.get("x_values");

    const wrapper = document.createElement("div");
    wrapper.style.fontFamily = "sans-serif";
    wrapper.style.fontSize = "13px";
    wrapper.style.maxWidth = "100%";
    // Keys pressed in the widget (space on a checkbox, say) must not reach
    // JupyterLab's notebook shortcuts.
    for (const kind of ["keydown", "keyup", "keypress"]) {
        wrapper.addEventListener(kind, (event) => event.stopPropagation());
    }

    // The figure, with a transparent canvas on top for the hover ring.
    const stage = document.createElement("div");
    stage.style.position = "relative";
    stage.style.display = "inline-block";
    stage.style.maxWidth = "100%";
    stage.style.transition = "opacity 0.1s";
    const img = document.createElement("img");
    img.style.display = "block";
    img.style.maxWidth = "100%";
    img.alt = "Data with the best-fit line, and the residuals underneath";
    const overlay = document.createElement("canvas");
    overlay.style.position = "absolute";
    overlay.style.left = "0";
    overlay.style.top = "0";
    overlay.style.pointerEvents = "none";
    const tip = document.createElement("div");
    tip.style.position = "absolute";
    tip.style.pointerEvents = "none";
    tip.style.background = "rgba(255, 255, 255, 0.95)";
    tip.style.border = "1px solid #bbbbbb";
    tip.style.padding = "2px 6px";
    tip.style.fontSize = "12px";
    tip.style.color = "#000000";
    tip.style.whiteSpace = "nowrap";
    tip.style.display = "none";
    stage.append(img, overlay, tip);

    const hint = document.createElement("div");
    hint.style.margin = "4px 0";
    hint.style.color = "var(--jp-ui-font-color1, #000000)";
    hint.textContent = "Click a point in either graph, or untick it in the table, to leave " +
                       "it out of the fit. Do the same again to put it back.";

    const message = document.createElement("div");
    message.style.margin = "4px 0";
    message.style.padding = "2px 6px";
    message.style.background = "var(--jp-warn-color3, #fff3cd)";
    message.style.display = "none";
    const say = (text) => {
        message.textContent = text;
        message.style.display = text ? "block" : "none";
    };

    const reset = document.createElement("button");
    reset.type = "button";
    reset.textContent = "Put all points back in the fit";
    reset.style.margin = "4px 0";
    reset.addEventListener("click", () => send([]));

    // ---- the selection --------------------------------------------------
    const excluded = () => new Set(model.get("excluded"));

    // Dimmed until Python sends the new figure; if that takes a while, say why.
    let waiting = null;
    function send(list) {
        say("");
        model.set("excluded", list.slice().sort((a, b) => a - b));
        model.save_changes();
        stage.style.opacity = "0.55";
        clearTimeout(waiting);
        waiting = setTimeout(() => say(
            "Waiting for Python to redraw the fit. If another cell is running, this " +
            "updates when it finishes. If the kernel was restarted, run this cell again."), 4000);
    }

    function toggle(k) {
        const out = excluded();
        if (out.has(k)) {
            out.delete(k);
        } else {
            const keep = [];
            for (let i = 1; i <= n; i++) if (!out.has(i) && i !== k) keep.push(i);
            if (keep.length < 3) {
                say(`Point ${k} cannot be left out: at least 3 points must stay in the fit.`);
                refresh();
                return;
            }
            if (new Set(keep.map((i) => xValues[i - 1])).size < 2) {
                say(`Point ${k} cannot be left out: the points left in the fit need at ` +
                    "least two different x values.");
                refresh();
                return;
            }
            out.add(k);
        }
        send([...out]);
    }

    // ---- the table of points ---------------------------------------------
    const checks = [];
    const rows = [];
    const table = document.createElement("div");
    table.style.maxHeight = "220px";
    table.style.overflowY = "auto";
    table.style.display = "inline-block";
    table.style.border = "1px solid #cccccc";
    table.style.margin = "4px 0";
    const grid = document.createElement("table");
    grid.style.borderCollapse = "collapse";
    const head = document.createElement("tr");
    for (const text of ["in fit", "point", "x", "y", "δy"]) {
        const th = document.createElement("th");
        th.textContent = text;
        th.style.padding = "2px 10px";
        th.style.textAlign = text === "in fit" ? "center" : "right";
        th.style.position = "sticky";
        th.style.top = "0";
        th.style.background = "var(--jp-layout-color2, #eeeeee)";
        head.append(th);
    }
    grid.append(head);
    for (let k = 1; k <= n; k++) {
        const tr = document.createElement("tr");
        const cell = document.createElement("td");
        cell.style.textAlign = "center";
        const check = document.createElement("input");
        check.type = "checkbox";
        check.setAttribute("aria-label", `point ${k} in the fit`);
        check.addEventListener("change", () => toggle(k));
        cell.append(check);
        tr.append(cell);
        for (const text of [String(k)].concat(points[k - 1])) {
            const td = document.createElement("td");
            td.textContent = text;
            td.style.padding = "1px 10px";
            td.style.textAlign = "right";
            td.style.fontVariantNumeric = "tabular-nums";
            tr.append(td);
        }
        tr.addEventListener("mouseenter", () => { hover = k; drawHover(); });
        tr.addEventListener("mouseleave", () => { hover = null; drawHover(); });
        checks.push(check);
        rows.push(tr);
        grid.append(tr);
    }
    table.append(grid);

    // ---- hover and click on the figure --------------------------------------
    let hover = null;                  // the point under the mouse or table row
    const scale = () => img.clientWidth / model.get("positions").width;

    function nearest(event) {
        const rect = img.getBoundingClientRect();
        const s = scale();
        const X = (event.clientX - rect.left) / s, Y = (event.clientY - rect.top) / s;
        const pos = model.get("positions");
        let best = null, bestDistance = 10 / s;       // within 10 screen pixels
        for (const panel of [pos.main, pos.resid]) {
            panel.forEach((place, i) => {
                if (!place) return;                   // not drawn in this graph
                const [px, py] = place;
                const d = Math.hypot(px - X, py - Y);
                if (d < bestDistance) { bestDistance = d; best = i + 1; }
            });
        }
        return best;
    }

    function drawHover() {
        const s = scale();
        const ratio = window.devicePixelRatio || 1;
        overlay.width = Math.round(img.clientWidth * ratio);
        overlay.height = Math.round(img.clientHeight * ratio);
        overlay.style.width = img.clientWidth + "px";
        overlay.style.height = img.clientHeight + "px";
        const ctx = overlay.getContext("2d");
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        ctx.clearRect(0, 0, img.clientWidth, img.clientHeight);
        rows.forEach((tr, i) => {
            tr.style.background = hover === i + 1 ? "rgba(255, 200, 0, 0.25)" : "";
        });
        if (hover === null || !model.get("positions").main.length) return;
        const pos = model.get("positions");
        ctx.strokeStyle = "#ff9900";
        ctx.lineWidth = 2;
        for (const panel of [pos.main, pos.resid]) {
            if (!panel[hover - 1]) continue;
            const [px, py] = panel[hover - 1];
            ctx.beginPath();
            ctx.arc(px * s, py * s, 8, 0, 2 * Math.PI);
            ctx.stroke();
        }
    }

    function showTip(k, event) {
        if (k === null) { tip.style.display = "none"; return; }
        const [x, y, dy] = points[k - 1];
        const state = excluded().has(k) ? "left out of the fit" : "in the fit";
        const action = excluded().has(k) ? " (click to put it back)" : " (click to leave it out)";
        tip.textContent = `point ${k}: x = ${x}, y = ${y} ± ${dy}, ${state}${action}`;
        const rect = stage.getBoundingClientRect();
        tip.style.display = "block";
        const left = event.clientX - rect.left + 14;
        tip.style.left = Math.min(left, rect.width - tip.offsetWidth - 2) + "px";
        tip.style.top = (event.clientY - rect.top + 14) + "px";
    }

    let lastMove = null;               // so the tip can be rewritten after a click
    img.addEventListener("mousemove", (event) => {
        lastMove = event;
        const k = nearest(event);
        img.style.cursor = k !== null ? "pointer" : "default";
        if (k !== hover) { hover = k; drawHover(); }
        showTip(k, event);
    });
    img.addEventListener("mouseleave", () => {
        lastMove = null; hover = null; drawHover(); tip.style.display = "none";
    });
    img.addEventListener("click", (event) => {
        const k = nearest(event);
        if (k !== null) toggle(k);
    });

    // ---- drawing what Python sent --------------------------------------------
    function refresh() {
        const out = excluded();
        checks.forEach((check, i) => { check.checked = !out.has(i + 1); });
        rows.forEach((tr, i) => { tr.style.color = out.has(i + 1) ? GREY : ""; });
        reset.disabled = out.size === 0;
    }

    function update() {
        clearTimeout(waiting);
        img.src = "data:image/png;base64," + model.get("png");
        img.style.width = model.get("positions").width + "px";
        stage.style.opacity = "1";
        say(model.get("message"));
        refresh();
        // The point under the mouse may now be in or out, or (an arrow in
        // the residuals graph) somewhere else: say so without waiting for a move.
        if (lastMove) {
            hover = nearest(lastMove);
            showTip(hover, lastMove);
        }
    }
    img.addEventListener("load", drawHover);
    const onUpdate = () => update();
    model.on("change:revision", onUpdate);

    const controls = document.createElement("div");
    controls.append(hint, message);
    controls.append(table);
    const resetRow = document.createElement("div");
    resetRow.append(reset);
    wrapper.append(stage, controls, resetRow);
    el.append(wrapper);
    update();

    return () => {
        clearTimeout(waiting);
        model.off("change:revision", onUpdate);
    };
}

export default { render };
"""

_autofit_widget_class = None


def _autofit_widget():
    """The widget class, built on first use so anywidget is only needed then."""
    global _autofit_widget_class
    if _autofit_widget_class is None:
        import anywidget
        import traitlets

        class AutofitWidget(anywidget.AnyWidget):
            _esm = _AUTOFIT_ESM
            n = traitlets.Int(0).tag(sync=True)
            points = traitlets.List([]).tag(sync=True)
            x_values = traitlets.List([]).tag(sync=True)
            excluded = traitlets.List([]).tag(sync=True)
            png = traitlets.Unicode("").tag(sync=True)
            positions = traitlets.Dict({"width": 1, "main": [], "resid": []}).tag(sync=True)
            revision = traitlets.Int(0).tag(sync=True)
            message = traitlets.Unicode("").tag(sync=True)

        _autofit_widget_class = AutofitWidget
    return _autofit_widget_class


class BestFit:
    """A best fit found by autofit, held in the notebook as autofit1,
    autofit2, and so on.

    .m and .b give the best-fit slope and intercept, .dm and .db their
    uncertainties, and .chi2 the reduced chi-squared, all worked out from
    the points in the fit, as plain floats. They follow your choices: leave
    a point out and they change.

      autofit1.m, autofit1.dm      the slope and its uncertainty
      autofit1.b, autofit1.db      the intercept and its uncertainty
      autofit1.chi2                the reduced chi-squared
      autofit1.excluded            the points left out of the fit, by number
                                   (the first point is 1)

    It also unpacks like a pair, so `m, b = autofit1` works. Shown on its
    own, it is the one-line summary autofit prints.
    """

    __slots__ = ("name", "_values", "_show_chi2", "_widget")

    def __init__(self, name, values, show_chi2):
        self.name = name
        self._values = values
        self._show_chi2 = show_chi2
        self._widget = None

    @property
    def m(self):
        """The best-fit slope."""
        return self._values["m"]

    @property
    def dm(self):
        """The uncertainty in the best-fit slope m."""
        return self._values["dm"]

    @property
    def b(self):
        """The best-fit intercept."""
        return self._values["b"]

    @property
    def db(self):
        """The uncertainty in the best-fit intercept b."""
        return self._values["db"]

    @property
    def chi2(self):
        """The reduced chi-squared of the best-fit line, over the points in
        the fit (divided by the number of them minus 2)."""
        return self._values["chi2"]

    @property
    def excluded(self):
        """The points left out of the fit, numbered from 1 for the first."""
        return list(self._values["excluded"])

    def __reduce__(self):
        # copy and pickle keep the values as they are now, without the widget
        return (BestFit, (self.name, dict(self._values), self._show_chi2))

    def __iter__(self):
        return iter((self.m, self.b))

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return (self.m, self.b)[index]

    def __repr__(self):
        return _autofit_line(self)


def _autofit_fit(x, y, dy, excluded):
    """The best fit to the points not in excluded (numbers from 1)."""
    keep = np.ones(len(x), dtype=bool)
    for k in excluded:
        keep[k - 1] = False
    m, b = _weighted_line_fit(x[keep], y[keep], dy[keep])
    dm, db = _line_fit_uncertainties(x[keep], dy[keep])
    chi2 = _reduced_chi2(x[keep], y[keep], dy[keep], m, b, int(np.sum(keep)) - 2)
    return {"m": m, "b": b, "dm": dm, "db": db, "chi2": chi2,
            "excluded": list(excluded), "used": int(np.sum(keep)), "n": len(x)}


_AUTOFIT_CSS_DPI = 100     # the picture's intended size: 100 px per inch
_AUTOFIT_SHARP = 2         # drawn at twice that, for high-resolution screens


def _autofit_draw(x, y, dy, values, labels, show_chi2, note=False):
    """Draws autofit's figure: plot_data's, with the points left out drawn
    hollow and grey.

    The residuals graph is scaled to the points in the fit. A point left
    out beyond it is shown as an arrow at the edge, with its number, which
    can be clicked; with note=True, a note at the top also counts these
    points.

    Returns (fig, ax_top, ax_bot, resid_shown), where resid_shown is where
    each residual is drawn (an off-scale point's arrow, for one beyond the
    scale). Drawn on a bare matplotlib Figure with the Agg canvas, never
    through pyplot, so no figure is registered, shown or left open.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.lines import Line2D

    grey = "#888888"
    keep = np.ones(len(x), dtype=bool)
    for k in values["excluded"]:
        keep[k - 1] = False
    out = ~keep
    m, b = values["m"], values["b"]
    with np.errstate(all='ignore'):
        resid = y - (m * x + b)

    # plot_data's 7 x 6 inch figure, with 0.4 inch more on the left, so that a
    # crop with a fixed left edge (in _autofit_figure) stays inside the
    # figure. The graphs are the same size as plot_data's.
    W, H = 7.4, 6
    fig = Figure(figsize=(W, H), dpi=_AUTOFIT_SHARP * _AUTOFIT_CSS_DPI)
    canvas = FigureCanvasAgg(fig)
    ax_top, ax_bot = fig.subplots(
        nrows=2, ncols=1, sharex=True,
        gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.08,
                     'left': (0.125 * 7 + 0.4) / W, 'right': (0.9 * 7 + 0.4) / W},
    )
    text_line = lambda label: Line2D([], [], linestyle="none", label=label)

    numbers = []                     # the left-out points' number labels
    data_artist = ax_top.errorbar(x[keep], y[keep], dy[keep], fmt='bo', markersize=3,
                                  label=labels["data_label"])
    handles = [data_artist]
    if np.any(out):
        handles.append(ax_top.errorbar(
            x[out], y[out], dy[out], fmt='o', markersize=5, mfc='none', mec=grey,
            ecolor=grey, label="Excluded from fit"))
        for k in values["excluded"]:
            numbers.append(ax_top.annotate(str(k), (x[k - 1], y[k - 1]), xytext=(6, 4),
                                           textcoords='offset points', color=grey, fontsize=9))
    # The line across all the data, so a point left out can be seen against it.
    x_model = np.linspace(np.min(x), np.max(x), 100)
    with np.errstate(all='ignore'):
        model_artist, = ax_top.plot(x_model, m * x_model + b, 'r-', label="Best fit (y=mx+b)")
    handles += [model_artist,
                text_line(_uncertainty_legend_text("m", m, values["dm"])),
                text_line(_uncertainty_legend_text("b", b, values["db"]))]
    if show_chi2:
        handles.append(text_line(_chi2_legend_text(values["chi2"])))
    ax_top.legend(handles=handles)
    ax_top.set_title(labels["title"])
    ax_top.set_ylabel(labels["y_label"])
    ax_top.grid(True, alpha=0.3)

    # The residuals graph is scaled to the points in the fit (error bars and
    # zero included), so that a point left out far from the line does not
    # squash the others. A point left out that falls outside is marked with
    # an arrow at the edge and, with note=True, counted in a note at
    # the top.
    lo = min(0.0, float(np.min(resid[keep] - dy[keep])))
    hi = max(0.0, float(np.max(resid[keep] + dy[keep])))
    span = hi - lo if hi > lo else 1.0
    above = out & (resid > hi + 0.05 * span)
    below = out & (resid < lo - 0.05 * span)
    inside = out & ~above & ~below
    note = bool(note) and bool(np.any(above | below))
    # Strips beyond the data, so that nothing covers a point: from the data
    # outwards, the note's strip (top only), then the arrows' strip at the
    # very edge, so an arrow never sits where a residual could be.
    lane = 0.18 * span                     # an arrow is about 10 px tall
    note_lane = 0.3 * span if note else 0.0
    y_top = hi + note_lane + (lane if np.any(above) else 0.05 * span)
    y_bottom = lo - (lane if np.any(below) else 0.05 * span)
    resid_shown = resid.copy()             # where each point is drawn, for clicking

    ax_bot.errorbar(x[keep], resid[keep], dy[keep], fmt='bo', markersize=3)
    if np.any(inside):
        ax_bot.errorbar(x[inside], resid[inside], dy[inside], fmt='o', markersize=5,
                        mfc='none', mec=grey, ecolor=grey)
    # A small arrow pointing off the graph, so it does not look like a
    # data point, with the point's number beside it. Its middle is the
    # place that is clicked.
    for side, sign, edge in ((above, 1, y_top - 0.5 * lane),
                             (below, -1, y_bottom + 0.5 * lane)):
        for i in np.flatnonzero(side):
            ax_bot.annotate("", xy=(x[i], edge + sign * 0.4 * lane),
                            xytext=(x[i], edge - sign * 0.4 * lane),
                            arrowprops={"arrowstyle": "-|>", "color": grey, "lw": 1.3,
                                        "mutation_scale": 9, "shrinkA": 0, "shrinkB": 0})
            numbers.append(ax_bot.annotate(str(i + 1), (x[i], edge), xytext=(5, 0),
                                           textcoords='offset points', color=grey,
                                           fontsize=9, va='center'))
            resid_shown[i] = edge
    if note:
        # A count, not the points' numbers, so it stays short however many
        # points are left out: "Off-scale points: 3 above, 1 below".
        counts = [f"{int(np.sum(side))} {where}"
                  for side, where in ((above, "above"), (below, "below")) if np.any(side)]
        if counts:
            # In the middle of its own strip, at the right.
            from matplotlib.transforms import blended_transform_factory
            ax_bot.text(0.99, hi + 0.5 * note_lane, "Off-scale points: " + ", ".join(counts),
                        transform=blended_transform_factory(ax_bot.transAxes, ax_bot.transData),
                        ha='right', va='center', color=grey,
                        fontsize=9, bbox={"facecolor": "white", "edgecolor": "#cccccc",
                                          "alpha": 0.9, "boxstyle": "round,pad=0.3"})
    ax_bot.hlines(y=0, xmin=np.min(x), xmax=np.max(x), color='k')
    if np.any(out):              # with every point in, the same scale as plot_data's
        ax_bot.set_ylim(y_bottom, y_top)
    ax_bot.set_xlabel(labels["x_label"])
    ax_bot.set_ylabel(RESIDUAL_LABEL, labelpad=10)
    ax_bot.grid(True, alpha=0.3)

    # Where points left out are close together, their numbers would run
    # into each other and into the next point ("1011121314"). Going left to
    # right, a number that would touch another number already shown, or
    # another left-out point's marker or arrow, is left off. The point is
    # still drawn (and its arrow can still be clicked), and the hover tip and
    # the table name it.
    if numbers:
        from matplotlib.transforms import Bbox
        for ax in (ax_top, ax_bot):
            ax.get_xlim(), ax.get_ylim()          # settle the axis ranges first
        renderer = canvas.get_renderer()
        pt = fig.dpi / 72                         # pixels per point
        def marker_box(ax, px, py, half_width, half_height):
            X, Y = ax.transData.transform((px, py))
            return Bbox.from_extents(X - half_width, Y - half_height,
                                     X + half_width, Y + half_height)
        markers = []                              # (label, the box its own point fills)
        for label in numbers:
            k = int(label.get_text()) - 1
            if label.axes is ax_top:
                markers.append((label, marker_box(ax_top, x[k], y[k], 3 * pt, 3 * pt)))
            else:
                bottom, top = ax_bot.transData.transform(
                    [(x[k], label.xy[1] - 0.4 * lane), (x[k], label.xy[1] + 0.4 * lane)])[:, 1]
                markers.append((label, marker_box(ax_bot, x[k], label.xy[1], 4 * pt,
                                                  (top - bottom) / 2)))
        shown = []
        for label, own in sorted(markers, key=lambda pair: pair[0].xy[0]):
            box = label.get_window_extent(renderer).padded(pt)
            others = [m for other, m in markers if other is not label and other.axes is label.axes]
            if any(box.overlaps(b) for b in shown + others):
                label.set_visible(False)
            else:
                shown.append(box)
    return fig, ax_top, ax_bot, resid_shown


def _autofit_figure(x, y, dy, values, labels, show_chi2, note=False,
                    left_edge=None):
    """autofit's figure (drawn by _autofit_draw), as a picture.

    left_edge (inches) is the furthest left the picture has been cropped
    before for this fit: the crop never starts to the right of it, so the
    graphs do not jump sideways when the tick labels narrow. The crop's
    left edge is returned in positions["left_edge"].

    Returns (PNG bytes, positions), where positions gives each point's place
    in the picture, in the picture's own pixels at its intended size, for
    the widget to find which point was clicked: {"width": ..., "height": ...,
    "main": [[X, Y], ...], "resid": [[X, Y], ...]}.
    """
    import io
    fig, ax_top, ax_bot, resid_shown = _autofit_draw(x, y, dy, values, labels,
                                                     show_chi2, note)
    canvas = fig.canvas
    W, H = fig.get_size_inches()
    sharp = _AUTOFIT_SHARP

    # Cropped to what is drawn, as plot_data's figures are in a notebook, so
    # long labels are never cut; the same crop decides the points' places.
    # Drawn once, and the drawing cropped, unless something reaches past the
    # figure's edge (a very long title, say); savefig then draws it again,
    # wide enough to hold everything.
    canvas.draw()
    dpi = fig.dpi
    crop = fig.get_tightbbox(canvas.get_renderer()).padded(0.1)     # inches
    # A fixed left edge, wide enough for most tick labels, and never right of
    # where this fit's picture started before: leaving a point out narrows
    # the residuals' tick labels, and the picture must not shift under the
    # mouse.
    from matplotlib.transforms import Bbox
    x0 = min(crop.x0, 0.1, left_edge if left_edge is not None else 0.1)
    crop = Bbox.from_extents(x0, crop.y0, crop.x1, crop.y1)
    buffer = io.BytesIO()
    if crop.x0 >= 0 and crop.y0 >= 0 and crop.x1 <= W and crop.y1 <= H:
        whole = np.asarray(canvas.buffer_rgba())
        height = whole.shape[0]
        left, right = int(np.floor(crop.x0 * dpi)), int(np.ceil(crop.x1 * dpi))
        top, bottom = int(np.floor(height - crop.y1 * dpi)), int(np.ceil(height - crop.y0 * dpi))
        top, left = max(top, 0), max(left, 0)
        from PIL import Image          # installed with matplotlib, which needs it
        Image.fromarray(np.ascontiguousarray(whole[top:bottom, left:right])).save(buffer, "png")
    else:
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches=crop)
        left, top = crop.x0 * dpi, (H - crop.y1) * dpi
        right, bottom = crop.x1 * dpi, (H - crop.y0) * dpi

    def places(ax, ys):
        shown = ax.transData.transform(np.column_stack([x, ys]))     # pixels, from bottom left
        # None for a point not drawn in this graph.
        return [None if not np.isfinite(py) else
                [round((float(px) - left) / sharp, 2),
                 round((H * dpi - float(py) - top) / sharp, 2)]
                for px, py in shown]

    positions = {
        "width": round((right - left) / sharp, 2),
        "height": round((bottom - top) / sharp, 2),
        "main": places(ax_top, y),
        "resid": places(ax_bot, resid_shown),
        "left_edge": float(crop.x0),
    }
    return buffer.getvalue(), positions


def _autofit_line(fit):
    """The fit's one-line summary: 'autofit1: m = 2.080 ± 0.056, ...'."""
    values = fit._values
    line = (f"{fit.name}: {_uncertainty_plain_text('m', values['m'], values['dm'])}, "
            f"{_uncertainty_plain_text('b', values['b'], values['db'])}")
    if fit._show_chi2:
        mantissa, exponent = _three_sig_figs(values["chi2"])
        line += f", χ² = {mantissa}" + ("" if exponent is None else f"×10^{exponent}")
    return line


def _autofit_summary(fit, pick=True):
    """The printed summary of a fit, as (plain text, HTML): two lines, the
    values, then which names hold them. Which points are left
    out is not repeated here: the table and the graph show it. Without
    pick, .excluded is not listed, since every point is in the fit.
    """
    line = _autofit_line(fit)
    rich =('<div style="font-family: sans-serif; font-size: 13px;">'
            f"<pre style='margin: 0;'>{html.escape(line)}</pre>")
    names = [f"{fit.name}.{v}" for v in ("m", "dm", "b", "db", "chi2", "excluded")
             if pick or v != "excluded"]
    plain = (line + "\n" + ", ".join(names[:-1]) + " and " + names[-1]
             + " hold these values at full precision.")
    code = lambda text: f"<code>{html.escape(text)}</code>"
    listed_html = ", ".join(code(n) for n in names[:-1]) + " and " + code(names[-1])
    rich += (f"<div style='margin-top: 4px;'>{listed_html} hold these values at full "
             "precision.</div></div>")
    return plain, rich


def _autofit_problem(x, excluded):
    """Why this choice of points cannot be fitted, or None."""
    keep = [i for i in range(len(x)) if i + 1 not in set(excluded)]
    if len(keep) < 3:
        return ("at least 3 points must stay in the fit, so that the residuals can "
                "show how well a line describes the data")
    if len({float(x[i]) for i in keep}) < 2:
        return "the points left in the fit need at least two different x values"
    return None


# Keywords that only change how a fit is shown.
_AUTOFIT_DISPLAY_KEYWORDS = ("note", "pick")


def _autofit_key(frame, x, y, dy):
    """What identifies a fit: its notebook cell and its call as typed, as for
    manual_fit, but without the keywords that only change how it is shown
    (note, pick), so that changing them keeps the fit's name and the points
    left out. Without the call's text, the data stand in for it."""
    cell = _current_cell_id()
    where = f"cell {cell} | " if cell else ""
    call = _typed_call('autofit', frame)
    if call is not None:
        try:
            call = ast.Call(func=call.func, args=call.args,
                            keywords=[k for k in call.keywords
                                      if k.arg not in _AUTOFIT_DISPLAY_KEYWORDS])
            return where + "call: " + ast.unparse(call)
        except Exception:
            pass
    return where + "autofit data: " + _autofit_data_text(x, y, dy)


def _autofit_data_text(x, y, dy):
    """The data as text, saved with the points left out: points are left out
    by number, so the choice is only used again on the same data. Otherwise
    an edit such as a reading added at the start would silently leave out a
    different point."""
    return json.dumps([x.tolist(), y.tolist(), dy.tolist()])


@_friendly_errors
def autofit(
    x, y, dy,
    title=None,
    x_label=None,
    y_label=None,
    data_label=None,
    chi2=True,
    note=False,
    pick=False,
):
    """Finds the straight line y = m*x + b that best fits your data, and
    plots it with the residuals underneath, as plot_data does with
    residuals=True. With pick=True, you can leave data points out of the
    fit and put them back.

    Arguments (positional order: x, y, dy):
      x  -- the x values of your data
      y  -- the y values of your data
      dy -- the uncertainty in each y value, one for every data point,
            none of them zero

    Needs at least 3 data points and at least two different x values.

    The best fit is the line with the smallest chi-squared. The legend gives
    its slope m and intercept b, each with its uncertainty, and the reduced
    chi-squared (divided by N - 2). Each uncertainty is given to 2
    significant figures, and its value to the same decimal place.

    You do not need to store anything. Each best fit is given a name:
    autofit1 for the first, autofit2 for the next, and so on. The name is
    printed under the graph with the values, as a line such as:

      autofit1: m = 2.080 ± 0.056, b = 0.005 ± 0.015, χ² = 0.663

    and in later cells it gives the values at full precision:

      autofit1.m, autofit1.dm      the slope and its uncertainty
      autofit1.b, autofit1.db      the intercept and its uncertainty
      autofit1.chi2                the reduced chi-squared
      autofit1.excluded            the points left out, by number (with
                                   pick=True)

    Leaving points out (with pick=True): click a point (in either graph) or
    untick it in the table under the graph; do the same again to put it
    back. The fit, the
    legend, the printed line and autofit1's values then use only the points
    left in. At least 3 points, with two different x values, must stay in
    the fit. Points are numbered from 1 for the first.

    Points left out are still shown, hollow and grey, with their number, so
    you can see how far they are from the line. The residuals graph is then
    scaled to the points in the fit, so that a point far off the line does
    not squash the others. A point left out that falls outside it is shown
    as a small arrow at the edge, with its number; click the arrow to put
    the point back. Where points left out are crowded together, some
    numbers are left off so they don't overlap; hover over a point or an
    arrow to see its number.

    Running the cell again, even after restarting the kernel, keeps the
    same name and leaves the same points out. If you have changed x, y or
    dy since, every point is back in the fit, and you choose again.
    Without pick=True, every point is in the fit; the points you chose are
    kept, and left out again when you add pick=True back.

    Optional:
      title, x_label, y_label, data_label -- as for plot_data. Without
                   x_label and y_label, the axes show the variable names
                   you typed for x and y.
      chi2      -- False to leave the reduced chi-squared out of the legend
                   and the printed line (autofit1.chi2 still holds it)
      note      -- True to add a note at the top of the residuals graph
                   saying how many points left out are off its scale,
                   e.g. "Off-scale points: 3 above, 1 below"
      pick      -- True to be able to leave points out of the fit and put
                   them back, by clicking them or with the table under the
                   graph

    Example:
      autofit(
          x=DxVec, y=FVec, dy=dFVec,
          title="Hooke's law investigation",
          x_label="Displacement of spring (m)",
          y_label="Force (N)",
      )

    x, y and dy can each be a list or a numpy array.
    """
    fn_name = 'autofit'
    for keyword, value in (("note", note), ("pick", pick)):
        if not isinstance(value, (bool, np.bool_)):
            _fail(fn_name,
                f"{fn_name}() needs {keyword} to be True or False.\n"
                f"  got {keyword} = {_shown(value)}"
            )
    pick = bool(pick)
    x, y, dy = _check_xy_data(x, y, dy, fn_name, zero_dy_is_error=True)
    if len(x) < 3:
        _fail(fn_name,
            f"{fn_name}() needs at least 3 data points, got {len(x)}.\n"
            "\n"
            "A straight line can always be drawn exactly through two points, so\n"
            "with fewer than three the residuals cannot show how well a line\n"
            "describes your data."
        )
    if np.all(x == x[0]):
        _fail(fn_name,
            f"{fn_name}() needs at least two different x values.\n"
            f"  all {len(x)} x values = {float(x[0]):g}\n"
            "\n"
            "A slope cannot be found when every point has the same x. Check\n"
            "that you passed the right array as x."
        )
    title, x_label, y_label, data_label, _ = _resolve_labels(
        fn_name, title, x_label, y_label, data_label, None,
    )
    labels = {"title": title, "x_label": x_label, "y_label": y_label,
              "data_label": data_label}

    try:
        widget_class = _autofit_widget() if pick else None
    except ImportError:
        _fail(fn_name,
            f"{fn_name}() needs the anywidget package, which is not\n"
            "installed in this Python environment.\n"
            "\n"
            "This is not a mistake in your code. Let your instructor or TA know."
        )

    # The fit's name, and the points left out: kept from an earlier run of
    # the same cell and call, in this kernel or saved beside the notebook.
    # The points left out are kept only if the data are unchanged; the name
    # is kept either way.
    frame = _student_frame()
    key = _autofit_key(frame, x, y, dy)
    data_text = _autofit_data_text(x, y, dy)
    path = _manual_fit_store_path(frame)
    stored = _manual_fit_store_read(path)
    if key in _autofits:
        name = _autofits[key].name
    else:
        names_in_use = {e.get("name") for e in stored.values() if isinstance(e, dict)}
        names_in_use |= {fit.name for fit in _autofits.values()}
        name = _manual_fit_name(key, stored, names_in_use, prefix="autofit")
    excluded = []
    entry = stored.get(key)
    if pick and isinstance(entry, dict) and entry.get("data") == data_text:
        saved = entry.get("excluded")
        if (isinstance(saved, list)
                and all(isinstance(k, int) and not isinstance(k, bool) and 1 <= k <= len(x)
                        for k in saved)
                and _autofit_problem(x, saved) is None):
            excluded = sorted(set(saved))

    values = _autofit_fit(x, y, dy, excluded)
    if any(_as_finite_number(values[v]) is None for v in ("m", "b", "dm", "db")):
        _fail(fn_name,
            f"{fn_name}() could not find a best-fit line for this data.\n"
            "\n"
            "Check that x, y and dy hold the values you meant, then run this\n"
            "cell again."
        )
    png, positions = _autofit_figure(x, y, dy, values, labels, chi2, note)

    if not pick:
        # Every point in the fit, and just the picture: no table, no
        # clicking. A choice saved with pick=True is kept as it is, for when
        # pick=True is added back; otherwise the name is saved, so it stays.
        fit = BestFit(name, values, bool(chi2))
        _autofits.pop(key, None)
        _autofits[key] = fit
        if not isinstance(entry, dict):
            fits = _manual_fit_store_read(path)
            fits[key] = {"name": name, "excluded": [], "data": data_text}
            _manual_fit_store_write(path, fits)
        if frame is not None:
            frame.f_globals[name] = fit
        plain, rich = _autofit_summary(fit, pick=False)
        pair = _notebook_display()
        if pair is None:
            print(plain)
            return
        display, _ = pair
        display({"image/png": png, "text/plain": f"{name}: a best fit made with {fn_name}"},
                raw=True, metadata={"image/png": {"width": int(round(positions["width"]))}})
        display({"text/plain": plain, "text/html": rich}, raw=True)
        return

    import base64
    widget = widget_class(
        n=len(x),
        points=[[f"{float(a):g}", f"{float(c):g}", f"{float(d):g}"] for a, c, d in zip(x, y, dy)],
        x_values=x.tolist(), excluded=excluded,
        png=base64.b64encode(png).decode("ascii"), positions=positions,
    )
    fit = BestFit(name, values, bool(chi2))
    fit._widget = widget
    _autofits.pop(key, None)
    _autofits[key] = fit                       # newest last

    def remember():
        fits = _manual_fit_store_read(path)
        fits[key] = {"name": name, "excluded": list(fit._values["excluded"]),
                     "data": data_text}
        _manual_fit_store_write(path, fits)

    remember()
    if frame is not None:
        frame.f_globals[name] = fit

    pair = _notebook_display()
    summary_handle = None

    def refit():
        chosen = sorted({int(k) for k in widget.excluded if 1 <= int(k) <= len(x)})
        problem = _autofit_problem(x, chosen)
        if problem:
            # The widget checks this first; this only guards against a
            # choice that reached Python some other way.
            return f"That choice cannot be fitted: {problem}."
        new_values = _autofit_fit(x, y, dy, chosen)
        if any(_as_finite_number(new_values[v]) is None for v in ("m", "b", "dm", "db")):
            return "No best-fit line could be found for the points left in."
        new_png, new_positions = _autofit_figure(x, y, dy, new_values, labels, chi2,
                                                   note, widget.positions["left_edge"])
        fit._values = new_values
        widget.positions = new_positions
        widget.png = base64.b64encode(new_png).decode("ascii")
        remember()
        if summary_handle is not None:
            plain, rich = _autofit_summary(fit)
            summary_handle.update({"text/plain": plain, "text/html": rich}, raw=True)
        return ""

    undoing = [False]

    def on_change(change):
        if undoing[0]:
            return
        # Whatever happens, the widget hears back (revision), so it never
        # waits for a figure that is not coming; a choice that could not be
        # fitted is undone, so the table shows the points actually fitted.
        try:
            message = refit()
        except Exception:
            message = ("Something went wrong redrawing the fit. Run this cell again; "
                       "if that does not help, let your instructor or TA know.")
        if message and list(widget.excluded) != fit._values["excluded"]:
            undoing[0] = True
            try:
                widget.excluded = list(fit._values["excluded"])
            finally:
                undoing[0] = False
        widget.message = message
        widget.revision += 1

    widget.observe(on_change, names=["excluded"])

    plain, rich = _autofit_summary(fit)
    if pair is None:
        print(plain)
        return
    display, _ = pair
    # One output holding the widget and a still picture, as for manual_fit:
    # a running notebook shows the widget; anything that cannot show a
    # widget falls back to the picture, which a re-run (as Restart Kernel and
    # Run All Cells does) brings up to date with the saved choice.
    bundle = widget._repr_mimebundle_()
    bundle = bundle[0] if isinstance(bundle, tuple) else bundle
    bundle["image/png"] = png
    bundle["text/plain"] = (f"{name}: a best fit made with {fn_name} "
                            "(the interactive plot appears in a running notebook)")
    # The picture is drawn at twice its intended size, for sharpness; the
    # width keeps it from showing twice as large where it is used.
    display(bundle, raw=True,
            metadata={"image/png": {"width": int(round(positions["width"]))}})
    # The summary is ordinary output, updated as points are left out or put
    # back, so it is kept when the notebook is saved or exported.
    summary_handle = display({"text/plain": plain, "text/html": rich}, raw=True,
                             display_id=True)
