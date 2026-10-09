# phys119-functions

Beginner-friendly measurement, uncertainty and plotting helpers for the UBC
Phys 119 lab course. The functions are written for students who are new to Python:
mistakes produce a plain-language message in red rather than a traceback,
and every function explains itself through `help()`.

## Functions

| Function | Returns |
|---|---|
| `t_score(x1, dx1, x2, dx2)` | the t'-score comparing two measurements |
| `mean(data)` | the mean of a set of measurements |
| `standard_deviation(data)` | the standard deviation of a set of repeated measurements |
| `standard_unc_of_mean(data)` | the standard uncertainty of the mean |
| `plot_data(x, y, dy, ...)` | a plot of data with error bars; with `m` and `b`, a straight-line model, and with `residuals=True`, the residuals underneath |
| `plot_residuals(x, y, dy, m, b, ...)` | the same as `plot_data(..., residuals=True)` |
| `autofit(x, y, dy, ...)` | the best-fit straight line (minimum chi-squared), plotted with its residuals; returns `m, b` |
| `manual_fit(x, y, dy, ...)` | an interactive fit in the notebook: sliders and number boxes for `m` and `b`, with the residuals updating live |

## Use in a notebook

```python
from phys119_functions import *

t_score(10, 2, 12, 3)                   # 0.5547
mean([10.1, 10.3, 9.8])                 # 10.07
standard_deviation([10.1, 10.3, 9.8])   # 0.2517
standard_unc_of_mean([10.1, 10.3, 9.8]) # 0.1453

help(t_score)                           # call signature and examples

plot_data(x=DxVec, y=FVec, dy=dFVec, m=2.1, b=0, residuals=True,
          title="Hooke's law", x_label=r"Compression $\Delta x$ (m)",
          y_label="Force (N)")
manual_fit(DxVec, FVec, dFVec)         # then manual_fit1.m, manual_fit1.b
```

That import also brings in `np`, as a backup for a notebook whose own
`import numpy as np` is missing. Lab notebooks should still carry that
import: it is what students are taught, and it is what makes their code work
outside the course. Both names refer to the same module, so having both costs
nothing, and a student who rebinds `np` in their notebook cannot affect these
functions.

## Design notes

- A **warning** appears as an amber block labelled `Warning:`. The
  calculation carries on and the result is still returned, because the values
  might be right.
- An **error** appears as a red block labelled `ERROR:` and stops the cell, by
  raising `Phys119Error`.
- Both blocks borrow Jupyter's own colour variables, so they sit in the
  colours the notebook already uses for machine output, and they are flat and
  square cornered rather than rounded with a left bar. Course notebooks use
  the rounded left-bar style for authored callouts, and the two should not
  look alike. Nothing further down the notebook runs on a result that was
  never produced. In a notebook the student sees the box and no traceback.
- Outside a notebook, both fall back to plain text on stderr, and the
  exception behaves like any other Python exception, which is what makes the
  test suite straightforward.
- Calling a function with the wrong number of arguments shows the usage line
  and a pointer to `help()`.
- Values come back as plain Python floats, so students see a clean number
  rather than `np.float64(...)`.
- Axis labels default to the variable names typed in the call, and titles
  and labels accept LaTeX between `$` signs. A LaTeX mistake gets a short
  message naming the label, rather than matplotlib's long error.
- `manual_fit` remembers each fit beside the notebook, so re-running the
  cell, even after a restart, brings the line back where it was left.

## Install

See [INSTALL.md](INSTALL.md), which covers installing for one account, for
every account on the Jupyter server, and using the module with no install at
all. Every [release](https://github.com/phys119/phys119_functions/releases)
carries its own wheel, so the shortest version is:

```
python -m pip install --no-deps \
    https://github.com/phys119/phys119_functions/releases/download/v1.3.0/phys119_functions-1.3.0-py3-none-any.whl
```

## Development

```
python -m pip install -e ".[test]"
python -m pytest
```

Bump `version` in `pyproject.toml` and `__version__` in the module together,
then rebuild:

```
python -m build
```

## Credit

Developed by Joss Ives for UBC Physics Labs, in collaboration with Claude
(Anthropic), 2026.

`standard_deviation` and `standard_unc_of_mean` are adapted from functions
developed by Rebeckah Fussell for Cornell Physics Labs.

MIT licensed. See LICENSE.
