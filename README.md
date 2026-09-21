# phys119-functions

Beginner-friendly measurement and uncertainty helpers for the UBC Phys 119
lab course. The functions are written for students who are new to Python:
mistakes produce a plain-language message in red rather than a traceback,
and every function explains itself through `help()`.

## Functions

| Function | Returns |
|---|---|
| `t_score(A, dA, B, dB)` | the t'-score comparing two measurements |
| `standard_deviation(data)` | the standard deviation of a set of repeated measurements |
| `standard_unc_of_mean(data)` | the standard uncertainty of the mean |

## Use in a notebook

```python
from phys119_functions import *

t_score(10, 2, 12, 3)                   # 0.5547
standard_deviation([10.1, 10.3, 9.8])   # 0.2517
standard_unc_of_mean([10.1, 10.3, 9.8]) # 0.1453

help(t_score)                           # call signature and examples
```

## Design notes

- Errors and warnings print to `sys.stderr`, which Jupyter shows in red.
  Errors return `None`; warnings still return the result.
- Nothing raises an exception at the student, and no traceback is shown.
- Calling a function with the wrong number of arguments prints the usage
  line and a pointer to `help()`.
- Values come back as plain Python floats, so students see a clean number
  rather than `np.float64(...)`.

## Install

For the course Jupyter server, as admin, from a built wheel:

```
python -m pip install phys119_functions-1.0.0-py3-none-any.whl
```

or directly from a checkout:

```
python -m pip install .
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
