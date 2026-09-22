# Changelog

## 1.1.0 (unreleased)

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

## 1.0.0 (deployed to the course JupyterHub 2026-09-22)

First packaged release, holding the three non-plotting helpers from the
course module: `t_score`, `standard_deviation` and `standard_unc_of_mean`.

- `standard_deviation` and `standard_unc_of_mean` are adapted from functions
  developed by Rebeckah Fussell for Cornell Physics Labs, with input
  checking and student-facing error messages added.
- `import *` exports only the three public functions.
- numpy is the only dependency. The plotting functions, and with them
  matplotlib and scipy, stay in the course copy of the module.
