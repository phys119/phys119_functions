# Installing phys119-functions

Three routes. Pick by who needs it.

| Route | Who it is for | Needs admin rights |
|---|---|---|
| [One account](#one-account) | a student, TA or colleague trying it out | no |
| [The whole Jupyter server](#the-whole-jupyter-server) | every account on the hub | yes |
| [No install at all](#no-install-at-all) | anyone with a notebook and no setup | no |

Replace `v1.3.0` and `1.3.0` throughout with the version you want. The
versions available are listed at
https://github.com/phys119/phys119_functions/releases.

## One account

In a notebook cell:

```
%pip install --user --no-deps --no-cache-dir https://github.com/phys119/phys119_functions/releases/download/v1.3.0/phys119_functions-1.3.0-py3-none-any.whl
```

**Restart the kernel afterwards** (Kernel > Restart Kernel). Python adds your
personal package folder to its search path only at startup, and on a first
install that folder is brand new, so the running kernel cannot see it.

Check what you got:

```python
import phys119_functions as pf

print(pf.__version__, pf.__file__)
```

A path under your home directory means the install worked. To remove it,
`%pip uninstall -y phys119-functions`, then restart the kernel again.

Two flags matter. `--user` keeps everything inside your home directory, so
nothing you do affects other accounts. `--no-deps` leaves the shared
packages alone. This package needs numpy 1.20 or newer and matplotlib, which
most hubs already have, and anywidget for `manual_fit` (see below).

## The whole Jupyter server

This needs a shell on the server and the right to install into the
environment all accounts share. On The Littlest JupyterHub that environment
is `/opt/tljh/user`; elsewhere, run `import sys; print(sys.executable)` in a
notebook to find the interpreter students are actually using, and substitute
it below.

```
sudo -H bash -c 'umask 022; /opt/tljh/user/bin/python -m pip install \
     --no-deps --no-cache-dir \
     https://github.com/phys119/phys119_functions/releases/download/v1.3.0/phys119_functions-1.3.0-py3-none-any.whl'

sudo /opt/tljh/user/bin/python -m pip check
ls -ld /opt/tljh/user/lib/python3.10/site-packages/phys119_functions*
```

Expect `No broken requirements found`, then `-rw-r--r--` on the module and
`drwxr-xr-x` on the `.dist-info` directory.

Three parts of that command are worth keeping:

- **`umask 022`.** Some systems give root a restrictive umask, in which case
  the installed files end up unreadable by anyone else. The install still
  reports success, and then every student gets `ModuleNotFoundError`. The
  permission listing above is what catches it.
- **`--no-deps`.** Nothing in a shared environment should move except this
  package. Without it, pip is free to upgrade numpy, which can break other
  packages pinned against the version that was there. Check the requirements
  separately instead: numpy 1.20 or newer, matplotlib, and anywidget.
- **The absolute interpreter path.** A bare `pip` depends on `PATH`, and over
  ssh that often resolves to the operating system's Python, where no student
  will ever see the result.

`manual_fit` also needs anywidget, which draws the fit in the notebook
through Jupyter's widget support (ipywidgets, which JupyterLab hubs
usually have). If the shared environment lacks it, install it the same
way, with `--no-deps`, after checking that its own requirements (psygnal,
typing_extensions and ipywidgets) are present. Without it, everything else
works and `manual_fit` says what is missing.

To roll back, run the same command with an earlier version in the URL.

Worth doing first on a single account, using the method above, and verifying
there before installing for everyone. `deploy/verify_install.ipynb` in this
repository runs that check and prints one pass or fail verdict.

## No install at all

Download the module and put it beside your notebook:

https://raw.githubusercontent.com/phys119/phys119_functions/v1.3.0/src/phys119_functions.py

Python imports a module from the notebook's own folder before any installed
one, so that file is what `from phys119_functions import *` will use. Nothing
else is needed beyond numpy, matplotlib and, for `manual_fit`, anywidget.

This is the easiest way to try a version without touching an installed copy,
and the easiest way to send the functions to someone who has nothing set up.

## Which copy am I using?

```python
import phys119_functions as pf

print(pf.__version__, pf.__file__)
```

| What the path shows | Which copy |
|---|---|
| your notebook's own folder | the single file sitting next to the notebook |
| `.local` under your home directory | installed for your account only |
| `/opt/tljh/user/...` | the shared install every account gets |

If a change you expected is missing, the module was almost certainly loaded
before that change arrived. Restart the kernel and check again.
