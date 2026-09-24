# Changelog

## 1.2.0 (deployed to the course JupyterHub 2026-09-23)

- `t_score` arguments are renamed from `A, dA, B, dB` to `x1, dx1, x2, dx2`,
  to match the notation the course materials use. A message reading "dx2 must
  be a number" points at something a student can find on the page in front of
  them, which "dB must be a number" did not. This changes keyword calls, and
  nothing in the course notebooks used them.
- The data functions gain their first two warnings, so a student meets the
  amber box on work of their own rather than only in a demonstration:
  - One measurement far from the others, which is usually a typing slip such
    as `4336` for `433.6`. Distance is measured from the median in units of
    the median absolute deviation, so one bad value cannot stretch the scale
    and hide itself, and the threshold is loose enough that ordinary lab data
    stays quiet. Worded as a prompt to check what was typed, never as a
    suggestion to discard a measurement. Applies to all three data functions.
  - Every measurement identical, so the spread is exactly zero. That usually
    means the instrument cannot resolve the repeats, which is a measurement
    problem rather than a Python one. Applies to the two spread functions; a
    mean of identical values is unremarkable and says nothing.
- Adds `mean(data)`, so the message a student gets for a mean matches the one
  they get for a standard deviation. It accepts a single measurement, since
  the mean of one number is that number, while the two spread functions still
  need two. Everything else it refuses, it refuses identically.
- Messages are restyled to look like machine output rather than like the
  course notebooks' authored callout boxes, which had come to share a visual
  language: pale fill, rounded corners, coloured left bar. A message is now
  flat, square cornered and unbordered, in Jupyter's own error and warning
  colours through `--jp-rendermime-error-background` and `--jp-warn-color3`,
  so it follows the theme.
- The error label is uppercase, `ERROR:` against `Warning:`, so the two differ
  by more than hue.
- `from phys119_functions import *` now also provides `np`, as a backup for a
  notebook missing its own `import numpy as np`. Notebooks should keep that
  import; this only stops a missing header cell from producing a `NameError`
  for a name students did not know the package needed.
- Docstring examples use plain lists rather than `np.array(...)`, so `help()`
  is copyable by a student who imported nothing else.
- Every message now names the function that produced it, written as `mean()`,
  and reads with that function as the subject: "mean() needs at least 1
  measurement, got 0", "t_score() needs x1 to be a number". Previously some
  messages named no function at all, which is ambiguous as soon as one cell
  calls more than one of them, and a demo cell that called two functions was
  read as attributing one function's message to another. A helper puts the
  name in front of anything that does not already start with it, so no message
  can reach a student unattributed.

## 1.1.0 (deployed to the course JupyterHub 2026-09-22)

Changes how mistakes are reported. Warnings and errors were previously
distinguishable only by their first word, and an early decision that errors
should be survivable was reversed.

- Warnings now appear as an amber box rather than sharing the red stderr
  styling with errors. The two were previously distinguishable only by their
  first word, which made a notebook hard to scan after Run All.
- Errors now stop the cell by raising `Phys119Error`, rather than printing
  and returning `None`. Nothing downstream runs on a result that was never
  produced, and the failing cell is the last one executed. In a notebook the
  student sees a red box and no traceback, through an IPython handler the
  module installs on import.
- `Phys119Error` is exported, so a notebook can catch it deliberately.
- Outside a notebook both kinds fall back to plain text on stderr.

Also fixed, found while re-checking the whole package:

- `nan` and `inf` arguments were accepted and travelled silently through the
  arithmetic, so `t_score(nan, 2, 12, 3)` returned `nan`. All four arguments
  must now be ordinary numbers.
- `nan` was caught in a dataset but `inf` was not, so
  `standard_deviation([1.0, inf])` returned `nan` and numpy printed its own
  `RuntimeWarning` naming a file path, which is the kind of message this
  package exists to avoid. Infinite values are now reported with their
  indices.
- A calculation that overflowed, such as `standard_deviation([1e200,
  -1e200])`, returned `inf` with another numpy warning. Calculations now run
  under `np.errstate` and a result that is not finite is reported instead of
  returned.
- An integer too large for a float raised a raw `OverflowError` traceback
  from `np.asarray`. It is now reported as a value too large to work with.
- Messages put the offending value on its own line, so one long input cannot
  produce a 123-character line, and the value is truncated at 60 characters.
- The exception message now carries the offending value alongside the
  headline, which is what a TA sees when catching the error or running a
  notebook non-interactively.

## 1.0.0 (deployed to the course JupyterHub 2026-09-22)

First packaged release, holding the three non-plotting helpers from the
course module: `t_score`, `standard_deviation` and `standard_unc_of_mean`.

- `standard_deviation` and `standard_unc_of_mean` are adapted from functions
  developed by Rebeckah Fussell for Cornell Physics Labs, with input
  checking and student-facing error messages added.
- `import *` exports only the three public functions.
- numpy is the only dependency. The plotting functions, and with them
  matplotlib and scipy, stay in the course copy of the module.
