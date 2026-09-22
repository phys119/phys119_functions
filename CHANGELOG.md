# Changelog

## 1.0.0 (unreleased)

First packaged release, holding the three non-plotting helpers from the
course module: `t_score`, `standard_deviation` and `standard_unc_of_mean`.

- `standard_deviation` and `standard_unc_of_mean` are adapted from functions
  developed by Rebeckah Fussell for Cornell Physics Labs, with input
  checking and student-facing error messages added.
- `import *` exports only the three public functions.
- numpy is the only dependency. The plotting functions, and with them
  matplotlib and scipy, stay in the course copy of the module.
