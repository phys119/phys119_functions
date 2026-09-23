"""Helper functions for Phys 119 lab notebooks.

Version 1.2.0. Four functions: t_score, mean, standard_deviation and
standard_unc_of_mean. This is the code installed on the course JupyterHub as
the package `phys119-functions`, and the same file can simply sit beside a
notebook, since Python imports a module from the notebook's own folder before
any installed one. Nothing needs installing, and numpy is all it needs.

The course folder holds a second copy under the same name which adds the
plotting functions. To tell them apart:
`hasattr(phys119_functions, 'plot_linear_model')` is False here and True
there.

Developed by Joss Ives for UBC Physics Labs, in collaboration with Claude
(Anthropic), 2026. The student-facing error messages, the input checking and
the usage help came out of that collaboration.

standard_deviation and standard_unc_of_mean are adapted from functions
developed by Rebeckah Fussell for Cornell Physics Labs.

Students import everything at the top of a lab notebook:

    from phys119_functions import *

and get help on any function with, for example, help(t_score).

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

import functools
import html
import sys
import inspect

import numpy as np

__version__ = "1.2.0"

__all__ = [
    "t_score",
    "mean",
    "standard_deviation",
    "standard_unc_of_mean",
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
_BOX = (
    '<div style="background:{background}; color:{colour};'
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

    shell.set_custom_exc((Phys119Error,), _handler)


_install_traceback_suppressor()


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
            # Python's own wording already opens with the function name, in
            # the same form these messages use, so it is left as it is.
            _fail(fn.__name__,
                f"{e}\n"
                f"\n"
                f"{usage}\n"
                f"Run help({fn.__name__}) for details."
            )
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

    return arr


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

    N = len(arr)
    with np.errstate(all='ignore'):
        unc = _sample_std(arr) / np.sqrt(N)
    return _finite_result(unc, 'standard_unc_of_mean')
