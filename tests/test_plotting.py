"""Tests for the plotting functions of phys119_functions: plot_data,
plot_residuals and autofit, and the label checks they share.

The first tests use the calls the course notebooks make, with the data from
the Prelab 05 notebook; later sections cover each feature as it was added.

Run with:  python -m pytest tests
"""

import inspect
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phys119_functions as p  # noqa: E402


# Prelab05_new.ipynb cell 23: spring data
DxVec = np.array([0.00, 0.12, 0.21, 0.29, 0.40, 0.48, 0.60])
FVec = np.array([0.00, 0.26, 0.43, 0.63, 0.81, 1.05, 1.23])
dFVec = np.array([0.02, 0.02, 0.03, 0.03, 0.03, 0.04, 0.04])

# Prelab05_new_error_demos.ipynb cell 1: shared sample data
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([2.1, 4.0, 6.2, 7.9, 10.1])
dy = np.array([0.2, 0.2, 0.3, 0.2, 0.3])

TITLE = "Hooke's law investigation using spring compression"
X_LABEL = "Displacement of spring from equilibrium (m)"
Y_LABEL = "Force (N)"


@pytest.fixture
def shown(monkeypatch):
    """Record each figure at the moment the function shows it."""
    figures = []
    monkeypatch.setattr(p.plt, "show", lambda *a, **k: figures.append(plt.gcf()))
    yield figures
    plt.close("all")


def errorbar_y(ax):
    """y values of the first errorbar data series on an axis."""
    return np.asarray(ax.containers[0].lines[0].get_ydata(), dtype=float)


def model_line(ax):
    """(x, y) of the red model line on an axis."""
    for line in ax.get_lines():
        if line.get_color() == "r":
            return (np.asarray(line.get_xdata(), dtype=float),
                    np.asarray(line.get_ydata(), dtype=float))
    raise AssertionError("no red model line found")


def legend_texts(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


def weighted_line_fit(xv, yv, sigma):
    """Weighted least-squares slope and intercept, the textbook formula."""
    w = 1 / sigma**2
    S, Sx, Sy = w.sum(), (w * xv).sum(), (w * yv).sum()
    Sxx, Sxy = (w * xv * xv).sum(), (w * xv * yv).sum()
    delta = S * Sxx - Sx**2
    return (S * Sxy - Sx * Sy) / delta, (Sxx * Sy - Sx * Sxy) / delta


# --- The calls Prelab05_new.ipynb makes ------------------------------------

def test_prelab_cell38_simplest_call(shown):
    p.plot_data(x=DxVec, y=FVec, dy=dFVec, m=2, b=0)
    assert len(shown) == 1
    (ax,) = shown[0].axes
    np.testing.assert_allclose(errorbar_y(ax), FVec)
    mx, my = model_line(ax)
    np.testing.assert_allclose(my, 2 * mx)
    assert (mx.min(), mx.max()) == (DxVec.min(), DxVec.max())


def test_prelab_cell40_labelled_simple(shown):
    p.plot_data(
        x=DxVec, y=FVec, dy=dFVec, m=2, b=0,
        title=TITLE, x_label=X_LABEL, y_label=Y_LABEL,
        data_label="Round 1 data", model_label="y = mx + b (model)",
    )
    (ax,) = shown[0].axes
    assert ax.get_title() == TITLE
    assert ax.get_xlabel() == X_LABEL
    assert ax.get_ylabel() == Y_LABEL
    assert legend_texts(ax) == ["Round 1 data", "y = mx + b (model)", "$\\mathrm{m} = 2.00$",
                                "$\\mathrm{b} = 0.00$"]


def test_prelab_cell43_residuals(shown):
    p.plot_data(
        x=DxVec, y=FVec, dy=dFVec, m=2, b=0,
        title=TITLE, x_label=X_LABEL, y_label=Y_LABEL,
        data_label="Round 1 data", model_label="y = mx + b (model)",
        residuals=True,
    )
    top, bottom = shown[0].axes
    assert top.get_title() == TITLE
    assert top.get_ylabel() == Y_LABEL
    assert bottom.get_xlabel() == X_LABEL
    assert bottom.get_ylabel() == "residual\n= data - model"
    np.testing.assert_allclose(errorbar_y(top), FVec)
    np.testing.assert_allclose(errorbar_y(bottom), FVec - 2 * DxVec)


def test_prelab_cell57_nonzero_slope(shown):
    p.plot_data(
        x=DxVec, y=FVec, dy=dFVec, m=2.1, b=0,
        title=TITLE, x_label=X_LABEL, y_label=Y_LABEL, data_label="Round 1 data",
        residuals=True,
    )
    top, bottom = shown[0].axes
    np.testing.assert_allclose(errorbar_y(bottom), FVec - 2.1 * DxVec)
    assert bottom.get_ylabel() == "residual\n= data - model"


def test_prelab_cell59_autofit(shown):
    p.autofit(
        x=DxVec, y=FVec, dy=dFVec,
        title=TITLE, x_label=X_LABEL, y_label=Y_LABEL,
    )
    top, bottom = shown[0].axes
    m_fit, b_fit = weighted_line_fit(DxVec, FVec, dFVec)
    model_entry = legend_texts(top)[0]
    assert model_entry == (
        f"y = mx + b (best fit)\nm = {m_fit:.4g}\nb = {b_fit:.4g}"
    )
    np.testing.assert_allclose(
        errorbar_y(bottom), FVec - (m_fit * DxVec + b_fit), atol=1e-12)


# --- Shape of the figures ---------------------------------------------------

def test_residuals_false_gives_one_panel(shown):
    p.plot_data(x=x, y=y, dy=dy, m=2.0, b=0.0, residuals=False)
    assert len(shown[0].axes) == 1


def test_intercept_used(shown):
    p.plot_data(x=x, y=y, dy=dy, m=2.0, b=0.5, residuals=True)
    top, bottom = shown[0].axes
    mx, my = model_line(top)
    np.testing.assert_allclose(my, 2.0 * mx + 0.5)
    np.testing.assert_allclose(errorbar_y(bottom), y - (2.0 * x + 0.5))


@pytest.mark.parametrize("call", [
    lambda: p.plot_data(x=x, y=y, dy=dy),
    lambda: p.plot_data(x=x, y=y, dy=dy, m=2.0),
    lambda: p.plot_data(x=x, y=y, dy=dy, m=2.0, residuals=True),
    lambda: p.autofit(x=x, y=y, dy=dy),
])
def test_no_figure_left_open(shown, call):
    call()
    assert len(shown) == 1
    assert plt.get_fignums() == []


# --- The cases Prelab05_new_error_demos.ipynb shows -------------------------

@pytest.mark.parametrize("call", [
    lambda: p.plot_data(x=x, y=y, dy=dy, m=2.0, b=0.0),
    lambda: p.plot_data(x=x, y=y, dy=dy, m=2.0, b=0.0),
    lambda: p.autofit(x=x, y=y, dy=dy),
])
def test_correct_call_is_silent(shown, capsys, call):
    call()
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("fn", ["plot_data"])
def test_missing_dy_shows_usage(shown, capsys, fn):
    with pytest.raises(p.Phys119Error):
        getattr(p, fn)(x=x, y=y)
    err = capsys.readouterr().err
    assert f"{fn}() missing 1 required positional argument: 'dy'" in err
    assert "Usage" in err
    assert shown == []


# --- plot_data: data, an optional line, optional residuals (2026-10-06) -----

def test_data_only_by_default(shown, capsys):
    p.plot_data(x, y, dy)
    (ax,) = shown[0].axes
    np.testing.assert_allclose(errorbar_y(ax), y)
    assert not [l for l in ax.get_lines() if l.get_color() == "r"]
    assert legend_texts(ax) == ["Data"]
    assert capsys.readouterr().err == ""


def test_m_alone_draws_a_line_through_the_origin(shown):
    p.plot_data(x, y, dy, m=2.0)
    (ax,) = shown[0].axes
    mx, my = model_line(ax)
    np.testing.assert_allclose(my, 2.0 * mx)


def test_residuals_without_a_line_is_an_error(shown, capsys):
    with pytest.raises(p.Phys119Error):
        p.plot_data(x, y, dy, residuals=True)
    assert capsys.readouterr().err.startswith(
        "ERROR: plot_data() needs a line to work out residuals from")
    assert shown == []


def test_intercept_without_slope_is_an_error(shown, capsys):
    with pytest.raises(p.Phys119Error):
        p.plot_data(x, y, dy, b=0.5)
    assert capsys.readouterr().err.startswith(
        "ERROR: plot_data() was given an intercept b but no slope m.")
    assert shown == []


def test_residual_label_is_fixed_and_clear_of_the_numbers(shown):
    p.plot_data(x, y, dy, m=2, residuals=True, y_label="Force (N)")
    bottom = shown[0].axes[1]
    assert bottom.get_ylabel() == "residual\n= data - model"
    assert bottom.yaxis.labelpad == 10


@pytest.mark.parametrize("fn", ["plot_data", "autofit"])
def test_length_mismatch_is_an_error(shown, capsys, fn):
    kwargs = {} if fn == "autofit" else {"m": 2.0}
    with pytest.raises(p.Phys119Error):
        getattr(p, fn)(x=x, y=y, dy=np.array([0.2, 0.2, 0.3]), **kwargs)
    err = capsys.readouterr().err
    assert err.startswith(f"ERROR: {fn}() needs x, y and dy to be the same length.")
    assert "len(x)=5, len(y)=5, len(dy)=3" in err
    assert shown == []


@pytest.mark.parametrize("fn", ["plot_data", "autofit"])
def test_negative_dy_is_an_error(shown, capsys, fn):
    kwargs = {} if fn == "autofit" else {"m": 2.0}
    with pytest.raises(p.Phys119Error):
        getattr(p, fn)(x=x, y=y, dy=np.array([0.2, -0.2, 0.3, 0.2, 0.3]), **kwargs)
    err = capsys.readouterr().err
    assert err.startswith(f"ERROR: {fn}() was given negative uncertainties in dy.")
    assert "indices: [1]" in err
    assert shown == []


@pytest.mark.parametrize("fn", ["plot_data"])
def test_zero_dy_warns_and_still_plots(shown, capsys, fn):
    getattr(p, fn)(x=x, y=y, dy=np.array([0.2, 0.0, 0.3, 0.2, 0.3]), m=2.0)
    err = capsys.readouterr().err
    assert err.startswith(f"Warning: {fn}() was given zeros in dy")
    assert "indices: [1]" in err
    assert len(shown) == 1


def test_zero_dy_stops_autofit(shown, capsys):
    with pytest.raises(p.Phys119Error):
        p.autofit(x=x, y=y, dy=np.array([0.2, 0.0, 0.3, 0.2, 0.3]))
    err = capsys.readouterr().err
    assert err.startswith("ERROR: autofit() was given zeros in dy")
    assert shown == []


# --- Step 1: the usage message is only for real call mistakes ---------------

@p._friendly_errors
def _stand_in(a, b=1):
    """A decorated function whose body raises a TypeError of its own."""
    return a + "text"


def test_internal_type_error_gets_no_usage_message(capsys):
    with pytest.raises(p.Phys119Error):
        _stand_in(5)
    err = capsys.readouterr().err
    assert err.startswith("ERROR: _stand_in() could not work with the values it was given.")
    assert "Python reported: unsupported operand" in err
    assert "Usage" not in err


@pytest.mark.parametrize("call", [
    lambda: _stand_in(),
    lambda: _stand_in(1, 2, 3),
    lambda: _stand_in(1, c=2),
])
def test_wrong_call_still_gets_usage_message(capsys, call):
    with pytest.raises(p.Phys119Error):
        call()
    err = capsys.readouterr().err
    assert "Usage (required): _stand_in(a)" in err
    assert "Usage (full):     _stand_in(a, b=1)" in err


# --- Step 2: one shared input check -----------------------------------------

ALL_THREE = ["plot_data", "autofit"]


def call(fn, xv, yv, dyv):
    kwargs = {} if fn == "autofit" else {"m": 2.0}
    return getattr(p, fn)(xv, yv, dyv, **kwargs)


@pytest.mark.parametrize("fn", ALL_THREE)
@pytest.mark.parametrize("kind", [list, tuple])
def test_lists_and_tuples_plot_the_same_as_arrays(shown, fn, kind):
    call(fn, x, y, dy)
    call(fn, kind(x), kind(y), kind(dy))
    as_array, as_other = shown
    for ax_a, ax_o in zip(as_array.axes, as_other.axes):
        np.testing.assert_allclose(errorbar_y(ax_o), errorbar_y(ax_a))
    np.testing.assert_allclose(model_line(as_other.axes[0])[1],
                               model_line(as_array.axes[0])[1])


@pytest.mark.parametrize("fn", ALL_THREE)
def test_single_number_dy_is_refused(shown, capsys, fn):
    with pytest.raises(p.Phys119Error):
        call(fn, x, y, 0.2)
    err = capsys.readouterr().err
    assert err.startswith(
        f"ERROR: {fn}() needs dy to hold one uncertainty for each data point.")
    assert "got the single number 0.2" in err
    assert shown == []


@pytest.mark.parametrize("fn", ALL_THREE)
@pytest.mark.parametrize("which", ["x", "y", "dy"])
def test_nan_is_an_error(shown, capsys, fn, which):
    data = {"x": x.copy(), "y": y.copy(), "dy": dy.copy()}
    data[which][3] = np.nan
    with pytest.raises(p.Phys119Error):
        call(fn, data["x"], data["y"], data["dy"])
    err = capsys.readouterr().err
    assert err.startswith(f"ERROR: {fn}() was given missing values (nan) in {which}.")
    assert "indices: [3]" in err
    assert shown == []


@pytest.mark.parametrize("fn", ALL_THREE)
@pytest.mark.parametrize("which", ["x", "y", "dy"])
def test_inf_is_an_error(shown, capsys, fn, which):
    data = {"x": x.copy(), "y": y.copy(), "dy": dy.copy()}
    data[which][1] = np.inf
    with pytest.raises(p.Phys119Error):
        call(fn, data["x"], data["y"], data["dy"])
    err = capsys.readouterr().err
    assert err.startswith(f"ERROR: {fn}() was given infinite values in {which}.")
    assert shown == []


@pytest.mark.parametrize("fn", ALL_THREE)
@pytest.mark.parametrize("bad, expected", [
    (5.0, "needs y to be a list or numpy array of values."),
    (["2.1", "4.0", "about 6", "7.9", "10.1"], "needs y to hold numbers only."),
    # numpy reads None as nan, so a blank entry gets the missing-value message
    ([2.1, 4.0, None, 7.9, 10.1], "was given missing values (nan) in y."),
    (y.reshape(-1, 1), "needs y to be a single list of values"),
    ([10**400] * 5, "was given a value in y too large to work with."),
])
def test_unusable_y_is_an_error(shown, capsys, fn, bad, expected):
    with pytest.raises(p.Phys119Error):
        call(fn, x, bad, dy)
    assert capsys.readouterr().err.startswith(f"ERROR: {fn}() {expected}")
    assert shown == []


@pytest.mark.parametrize("fn", ALL_THREE)
def test_empty_data_is_an_error(shown, capsys, fn):
    with pytest.raises(p.Phys119Error):
        call(fn, [], [], [])
    assert capsys.readouterr().err.startswith(
        f"ERROR: {fn}() was given no data: x, y and dy are all empty.")
    assert shown == []


@pytest.mark.parametrize("fn", ALL_THREE)
def test_no_raw_numpy_warning_reaches_a_student(shown, fn):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        call(fn, list(x), list(y), list(dy))


# --- Step 3: m and b must be ordinary numbers --------------------------------

PLOTS = ["plot_data"]


@pytest.mark.parametrize("fn", PLOTS)
@pytest.mark.parametrize("params, expected", [
    ({"m": "2"}, "needs m to be a number.\n  got str: '2'"),
    ({"m": [2]}, "needs m to be a number.\n  got list: [2]"),
    ({"m": 2, "b": "0.5"}, "needs b to be a number.\n  got str: '0.5'"),
    ({"m": 2, "b": np.nan}, "needs b to be an ordinary number.\n  got nan"),
    ({"m": np.inf}, "needs m to be an ordinary number.\n  got inf"),
])
def test_unusable_m_or_b_is_an_error(shown, capsys, fn, params, expected):
    with pytest.raises(p.Phys119Error):
        getattr(p, fn)(x, y, dy, **params)
    assert capsys.readouterr().err.startswith(f"ERROR: {fn}() {expected}")
    assert shown == []


@pytest.mark.parametrize("fn", PLOTS)
def test_quoted_number_gets_a_hint(shown, capsys, fn):
    with pytest.raises(p.Phys119Error):
        getattr(p, fn)(x, y, dy, m=" 2.1 ")
    err = capsys.readouterr().err
    assert "Write it\nwithout them: m=2.1" in err


@pytest.mark.parametrize("fn", PLOTS)
def test_unquotable_text_gets_no_hint(shown, capsys, fn):
    with pytest.raises(p.Phys119Error):
        getattr(p, fn)(x, y, dy, m="two")
    assert "quotation marks" not in capsys.readouterr().err


@pytest.mark.parametrize("fn", PLOTS)
@pytest.mark.parametrize("m, b", [(2, 0), (2.0, -1.5), (np.float64(2.0), np.int64(1))])
def test_numeric_types_accepted(shown, capsys, fn, m, b):
    getattr(p, fn)(x, y, dy, m=m, b=b)
    assert capsys.readouterr().err == ""
    assert len(shown) == 1


# --- Step 4: axis labels come from the names the student typed ---------------

def axis_labels(fig):
    return fig.axes[-1].get_xlabel(), fig.axes[0].get_ylabel()


def test_typed_names_become_axis_labels(shown):
    # Spelled out: the name is read from the call as typed, so a call made
    # through getattr() would (correctly) fall back.
    p.plot_data(x=DxVec, y=FVec, dy=dFVec, m=2)
    p.plot_data(DxVec, FVec, dFVec, m=2)
    p.autofit(DxVec, FVec, dFVec)
    assert [axis_labels(f) for f in shown] == [("DxVec", "FVec")] * 3


def test_call_over_several_lines(shown):
    p.plot_data(
        x=DxVec,
        y=FVec,
        dy=dFVec,
        m=2,
    )
    assert axis_labels(shown[0]) == ("DxVec", "FVec")


def test_alias_shows_the_name_typed(shown):
    x_values = DxVec
    p.plot_data(x_values, FVec, dFVec, m=2)
    assert axis_labels(shown[0]) == ("x_values", "FVec")


def test_anything_but_a_plain_name_falls_back(shown):
    p.plot_data(list(DxVec), FVec * 100, dFVec, m=2)
    assert axis_labels(shown[0]) == ("x (default name)", "y (default name)")


def test_two_calls_on_one_line_fall_back(shown):
    p.plot_data(DxVec, FVec, dFVec, m=2); p.plot_data(FVec, DxVec, dFVec, m=2)  # noqa: E702
    assert [axis_labels(f) for f in shown] == [("x (default name)", "y (default name)")] * 2


def test_given_labels_win(shown):
    p.plot_data(DxVec, FVec, dFVec, m=2, x_label=X_LABEL)
    assert axis_labels(shown[0]) == (X_LABEL, "FVec")


def test_label_lookup_never_fails_a_call(shown, monkeypatch):
    monkeypatch.setattr(p, "_student_frame", lambda: None)
    p.plot_data(DxVec, FVec, dFVec, m=2)
    assert axis_labels(shown[0]) == ("x (default name)", "y (default name)")


def test_labels_in_an_ipython_cell(shown):
    """The way a notebook runs it: cell source held by IPython, not a file."""
    from IPython.core.interactiveshell import InteractiveShell
    shell = InteractiveShell.instance()
    shell.user_ns.update(p=p, DxVec=DxVec, FVec=FVec, dFVec=dFVec)
    shell.run_cell("DxVec", silent=True)   # also stores it as _ and _1
    shell.run_cell("p.autofit(\n    x=DxVec,\n    y=FVec,\n    dy=dFVec,\n)",
                   silent=True)
    shell.run_cell("x = DxVec\np.plot_data(x, FVec, dFVec, m=2)", silent=True)
    assert [axis_labels(f) for f in shown] == [("DxVec", "FVec"), ("x", "FVec")]


# --- Step 5: autofit refuses data a line fit cannot judge -------------

@pytest.mark.parametrize("n", [1, 2])
def test_autofit_needs_three_points(shown, capsys, n):
    with pytest.raises(p.Phys119Error):
        p.autofit(x[:n], y[:n], dy[:n])
    assert capsys.readouterr().err.startswith(
        f"ERROR: autofit() needs at least 3 data points, got {n}.")
    assert shown == []


def test_autofit_accepts_three_points(shown, capsys):
    p.autofit(x[:3], y[:3], dy[:3])
    assert capsys.readouterr().err == ""
    assert len(shown) == 1


def test_autofit_needs_two_different_x_values(shown, capsys):
    with pytest.raises(p.Phys119Error):
        p.autofit([2.0] * 4, y[:4], dy[:4])
    err = capsys.readouterr().err
    assert err.startswith("ERROR: autofit() needs at least two different x values.")
    assert "all 4 x values = 2" in err
    assert shown == []


def test_autofit_accepts_repeated_x_values(shown, capsys):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p.autofit([1.0, 1.0, 2.0], [2.0, 2.2, 4.1], [0.1, 0.1, 0.1])
    assert capsys.readouterr().err == ""
    assert len(shown) == 1


def test_autofit_reports_a_failed_fit(shown, capsys, monkeypatch):
    monkeypatch.setattr(p, "_weighted_line_fit", lambda *args: (float("nan"), float("nan")))
    with pytest.raises(p.Phys119Error):
        p.autofit(x, y, dy)
    assert capsys.readouterr().err.startswith(
        "ERROR: autofit() could not find a best-fit line for this data.")
    assert shown == []


@pytest.mark.parametrize("xv, yv, sv, slope", [
    ([1e200, 2e200, 3e200], [1.0, 2.0, 3.1], [0.1] * 3, 1.05e-200),   # squares would overflow
    ([1.0, 2.0, 3.0], [1.0, 2.0, 3.1], [1e-200] * 3, 1.05),           # 1/dy^2 would overflow
])
def test_extreme_values_give_the_right_line_not_zero(shown, xv, yv, sv, slope):
    m, b = p._weighted_line_fit(np.array(xv), np.array(yv), np.array(sv))
    assert m == pytest.approx(slope, rel=1e-9)


@pytest.mark.parametrize("xv, yv, sv, expected", [
    # data exactly on a line: the line itself, with no rounding noise in b
    ([0.0, 1, 2, 3, 4], [1.0, 3, 5, 7, 9], [0.1] * 5, (2.0, 1.0)),
    # x far from zero (years): still exact
    ([2000.0, 2001, 2002, 2003, 2004], [5.0, 5.2, 5.5, 5.6, 5.9], [0.1] * 5, (0.22, -435.0)),
])
def test_autofit_is_the_exact_chi_squared_minimum(shown, xv, yv, sv, expected):
    fit = p.autofit(xv, yv, sv)
    assert fit.m == pytest.approx(expected[0], rel=1e-12)
    assert fit.b == pytest.approx(expected[1], rel=1e-12, abs=1e-12)


def test_autofit_weights_points_by_their_uncertainty(shown):
    # Two tight points at y = x and one loose point far off the line: the
    # best fit stays close to y = x.
    fit = p.autofit([0.0, 1.0, 2.0], [0.0, 1.0, 5.0], [0.01, 0.01, 10.0])
    assert fit.m == pytest.approx(1.0, abs=1e-3)
    assert fit.b == pytest.approx(0.0, abs=1e-3)


# --- Step 6: autofit returns the best fit ------------------------------

def test_autofit_returns_m_and_b(shown):
    m, b = p.autofit(DxVec, FVec, dFVec)
    m_exact, b_exact = weighted_line_fit(DxVec, FVec, dFVec)
    assert m == pytest.approx(m_exact, abs=1e-6)
    assert b == pytest.approx(b_exact, abs=1e-6)
    assert type(m) is float and type(b) is float


def test_result_offers_m_and_b_by_name(shown):
    fit = p.autofit(DxVec, FVec, dFVec)
    assert (fit.m, fit.b) == tuple(fit)


def test_result_displays_as_one_tidy_line(shown):
    fit = p.autofit(DxVec, FVec, dFVec)
    assert repr(fit) == f"best fit: m = {fit.m:.4g}, b = {fit.b:.4g}"
    assert str(fit) == repr(fit)


def test_result_line_matches_the_legend(shown):
    fit = p.autofit(DxVec, FVec, dFVec)
    assert legend_texts(shown[0].axes[0])[0] == (
        f"y = mx + b (best fit)\nm = {fit.m:.4g}\nb = {fit.b:.4g}")


def test_notebook_shows_the_tidy_line(shown):
    from IPython.core.interactiveshell import InteractiveShell
    shell = InteractiveShell.instance()
    fit = p.autofit(DxVec, FVec, dFVec)
    shown_text = shell.display_formatter.format(fit)[0]["text/plain"]
    assert shown_text == repr(fit)


# --- Step 7: internal tidying --------------------------------------------------

def test_default_sizes_unchanged(shown):
    p.plot_data(x, y, dy)
    p.plot_data(x, y, dy, m=2)
    p.plot_data(x, y, dy, m=2, residuals=True)
    p.autofit(x, y, dy)
    p.autofit(x, y, dy, residuals=False)
    default = tuple(plt.rcParams["figure.figsize"])
    sizes = [tuple(f.get_size_inches()) for f in shown]
    assert sizes == [default, default, (7, 6), (7, 6), default]


@pytest.mark.parametrize("residuals", [True, False])
def test_figsize_honoured_with_and_without_residuals(shown, residuals):
    p.plot_data(x, y, dy, m=2, residuals=residuals, figsize=(5, 8))
    p.autofit(x, y, dy, residuals=residuals, figsize=(5, 8))
    assert [tuple(f.get_size_inches()) for f in shown] == [(5, 8), (5, 8)]


def test_extra_positional_argument_cannot_skip_the_checks(shown, capsys):
    with pytest.raises(p.Phys119Error):
        p.plot_data(x, y, [-1] * 5, 2, 0, None, None, None, None, None,
                    True, 100, None, False, True)
    err = capsys.readouterr().err
    assert "takes from 3 to 14 positional arguments but 15 were given" in err
    assert "_suppress_checks" not in err
    assert shown == []


# --- Step 8: the help text -----------------------------------------------------

def test_help_example_result_line_is_the_real_fit(shown):
    """The line quoted in help(autofit) is what the Prelab data give."""
    fit = p.autofit(x=DxVec, y=FVec, dy=dFVec)
    assert f"line such as:  {fit!r}" in p.autofit.__doc__


@pytest.mark.parametrize("fn", ALL_THREE)
def test_help_hides_private_parameters(fn):
    import pydoc
    text = pydoc.render_doc(getattr(p, fn), renderer=pydoc.plaintext)
    assert "_suppress_checks" not in text
    assert "scipy" not in text and "curve_fit" not in text


@pytest.mark.parametrize("fn", ALL_THREE)
def test_help_lists_every_required_argument(fn):
    doc = inspect.getdoc(getattr(p, fn))
    for name in ("x", "y", "dy"):
        assert f"\n  {name:<2} -- " in doc


# --- Two copies of the module in one notebook -----------------------------------

def test_earlier_copy_keeps_its_traceback_suppression():
    """Importing a second copy (say a dev copy beside the installed package)
    must not make the first copy's errors show full tracebacks again."""
    from IPython.core.interactiveshell import InteractiveShell
    shell = InteractiveShell.instance()
    EarlierCopy = type("Phys119Error", (Exception,), {})
    shell.set_custom_exc((EarlierCopy,), lambda *a, **k: [])
    p._install_traceback_suppressor()
    assert set(shell.custom_exceptions) == {EarlierCopy, p.Phys119Error}
    p._install_traceback_suppressor()          # importing again adds nothing
    assert set(shell.custom_exceptions) == {EarlierCopy, p.Phys119Error}
    shell.set_custom_exc((p.Phys119Error,), lambda *a, **k: [])


def test_unrelated_registration_is_not_silenced():
    from IPython.core.interactiveshell import InteractiveShell
    shell = InteractiveShell.instance()
    shell.set_custom_exc((KeyError,), lambda *a, **k: None)
    p._install_traceback_suppressor()
    assert shell.custom_exceptions == (p.Phys119Error,)


# --- Review fixes, 2026-10-05 ---------------------------------------------------

def test_error_is_checked_before_any_warning(shown, capsys):
    """A zero in dy (a warning) together with a bad m (an error) must show the
    error alone, not an amber box for a plot that is never drawn."""
    with pytest.raises(p.Phys119Error):
        p.plot_data(x, y, [0.2, 0.0, 0.3, 0.2, 0.3], m="two")
    err = capsys.readouterr().err
    assert err.startswith("ERROR: plot_data() needs m to be a number.")
    assert "Warning" not in err


@pytest.mark.parametrize("fn", ALL_THREE)
@pytest.mark.parametrize("bad", [-1, 0, 1, 2.5, "100", True])
def test_num_model_points_must_be_a_whole_number(shown, capsys, fn, bad):
    kwargs = {} if fn == "autofit" else {"m": 2.0}
    with pytest.raises(p.Phys119Error):
        getattr(p, fn)(x, y, dy, num_model_points=bad, **kwargs)
    assert capsys.readouterr().err.startswith(
        f"ERROR: {fn}() needs num_model_points to be a whole number, 2 or more.")


def test_num_model_points_accepts_numpy_integers(shown):
    p.plot_data(x, y, dy, m=2, num_model_points=np.int64(50))
    assert len(model_line(shown[0].axes[0])[0]) == 50


@pytest.mark.parametrize("dy_value, got", [
    (np.array(0.2), "got the single number array(0.2)"),
    (None, "got NoneType: None"),
])
def test_other_single_values_for_dy(shown, capsys, dy_value, got):
    with pytest.raises(p.Phys119Error):
        p.plot_data(x, y, dy_value, m=2)
    err = capsys.readouterr().err
    assert err.startswith("ERROR: plot_data() needs dy to hold one uncertainty")
    assert got in err


def test_ragged_data_is_an_error_not_a_traceback(shown, capsys):
    with pytest.raises(p.Phys119Error):
        p.plot_data(x, [[1, 2], [3]], dy, m=2)
    assert capsys.readouterr().err.startswith(
        "ERROR: plot_data() needs y to hold numbers only.")


def test_whole_fit_passed_as_m_gets_a_hint(shown, capsys):
    fit = p.autofit(x, y, dy)
    capsys.readouterr()
    with pytest.raises(p.Phys119Error):
        p.plot_data(x, y, dy, m=fit)
    err = capsys.readouterr().err
    assert "needs m to be a number" in err
    assert "m, b = autofit(...), or use fit.m and fit.b." in err


def test_best_fit_survives_copy_and_pickle(shown):
    import copy
    import pickle
    fit = p.autofit(x, y, dy)
    for clone in (copy.copy(fit), copy.deepcopy(fit), pickle.loads(pickle.dumps(fit))):
        assert clone == fit and type(clone) is p.BestFit and clone.m == fit.m


def test_huge_values_leak_no_numpy_warning(shown):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p.plot_data(x * 1e10, y, dy, m=1e300)


def test_label_lookup_survives_any_failure(shown, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("anything at all")
    monkeypatch.setattr(p.linecache, "getlines", broken)
    p.plot_data(DxVec, FVec, dFVec, m=2)
    assert axis_labels(shown[0]) == ("x (default name)", "y (default name)")


def test_autofit_needs_no_scipy(shown, monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.optimize", None)
    m, b = p.autofit(DxVec, FVec, dFVec)
    assert (m, b) == pytest.approx(weighted_line_fit(DxVec, FVec, dFVec), rel=1e-12)


# --- The 2026-10-05 rename ------------------------------------------------------

@pytest.mark.parametrize("old", ["plot_linear_model_simple", "plot_linear_model",
                                 "autofit_simple", "plot_linear_simple", "plot_linear",
                                 "autofit_linear", "manual_fit_linear"])
def test_old_names_are_gone(old):
    assert not hasattr(p, old)
    assert old not in p.__all__


def test_no_message_or_help_text_uses_an_old_name():
    import inspect as _inspect
    source = _inspect.getsource(p)
    for old in ("plot_linear_model", "autofit_simple", "plot_linear",
                "autofit_linear", "manual_fit_linear", "residual_y_label="):
        assert old not in source


# --- plot_residuals: plot_data with residuals=True (2026-10-06) -----------------

def test_plot_residuals_draws_what_plot_data_with_residuals_draws(shown):
    labels = dict(title=TITLE, x_label=X_LABEL, y_label=Y_LABEL, data_label="Round 1 data")
    p.plot_data(DxVec, FVec, dFVec, m=2.1, b=0.01, residuals=True, **labels)
    p.plot_residuals(DxVec, FVec, dFVec, m=2.1, b=0.01, **labels)
    via_data, via_residuals = shown
    assert len(via_residuals.axes) == 2
    for a, r in zip(via_data.axes, via_residuals.axes):
        np.testing.assert_allclose(errorbar_y(r), errorbar_y(a))
        assert (r.get_title(), r.get_xlabel(), r.get_ylabel()) == \
               (a.get_title(), a.get_xlabel(), a.get_ylabel())
    np.testing.assert_allclose(model_line(via_residuals.axes[0])[1],
                               model_line(via_data.axes[0])[1])


def test_plot_residuals_b_defaults_to_zero(shown):
    p.plot_residuals(x, y, dy, m=2.0)
    np.testing.assert_allclose(errorbar_y(shown[0].axes[1]), y - 2.0 * x)


def test_plot_residuals_needs_m(shown, capsys):
    with pytest.raises(p.Phys119Error):
        p.plot_residuals(x, y, dy)
    err = capsys.readouterr().err
    assert "plot_residuals() missing 1 required positional argument: 'm'" in err
    assert "Usage (required): plot_residuals(x, y, dy, m)" in err
    assert shown == []


def test_plot_residuals_messages_name_plot_residuals(shown, capsys):
    with pytest.raises(p.Phys119Error):
        p.plot_residuals(x, y, dy=0.2, m=2)
    assert capsys.readouterr().err.startswith(
        "ERROR: plot_residuals() needs dy to hold one uncertainty")
    with pytest.raises(p.Phys119Error):
        p.plot_residuals(x, y, dy, m=None)
    err = capsys.readouterr().err
    assert err.startswith("ERROR: plot_residuals() needs a line")
    assert "plot_residuals(x, y, dy, m=2, b=0)" in err and "residuals=True" not in err


def test_plot_residuals_labels_axes_with_the_typed_names(shown):
    p.plot_residuals(DxVec, FVec, dFVec, m=2)
    assert axis_labels(shown[0]) == ("DxVec", "FVec")


def test_plot_residuals_has_no_residuals_keyword(shown, capsys):
    with pytest.raises(p.Phys119Error):
        p.plot_residuals(x, y, dy, m=2, residuals=False)
    assert "unexpected keyword argument 'residuals'" in capsys.readouterr().err


def test_plot_residuals_exported_and_documented():
    assert "plot_residuals" in p.__all__
    assert "plot_residuals(x, y, dy, m, b) does the same." in p.plot_data.__doc__
    for name in ("x", "y", "dy", "m", "b"):
        assert f"\n  {name:<2} -- " in inspect.getdoc(p.plot_residuals)



# --- Reduced chi-squared, chi2=True (2026-10-06) ---------------------------------

def chi2_by_hand(xv, yv, dyv, m, b, dof):
    xv, yv, dyv = map(np.asarray, (xv, yv, dyv))
    return np.sum(((yv - (m * xv + b)) / dyv) ** 2) / dof


def legend_chi2(ax):
    texts = legend_texts(ax)
    assert texts[-1].startswith("$\\chi^2 = "), texts
    return texts


@pytest.mark.parametrize("value, text", [
    (1.2, "$\\chi^2 = 1.20$"),
    (0.0456, "$\\chi^2 = 0.0456$"),
    (123.4, "$\\chi^2 = 123$"),
    (999.6, "$\\chi^2 = 1.00 \\times 10^{3}$"),
    (1234, "$\\chi^2 = 1.23 \\times 10^{3}$"),
    (0.000123, "$\\chi^2 = 1.23 \\times 10^{-4}$"),
    (0.00123, "$\\chi^2 = 0.00123$"),
    (0, "$\\chi^2 = 0.00$"),
])
def test_chi2_is_written_to_three_significant_figures(value, text):
    assert p._chi2_legend_text(value) == text


def test_chi2_off_by_default(shown):
    p.plot_data(DxVec, FVec, dFVec, m=2, b=0.01)
    assert legend_texts(shown[0].axes[0]) == ["Data", "Model (y=mx+b)",
                                              "$\\mathrm{m} = 2.00$", "$\\mathrm{b} = 0.0100$"]


def test_chi2_is_the_last_legend_line(shown):
    p.plot_data(DxVec, FVec, dFVec, m=2, b=0.01, chi2=True)
    texts = legend_chi2(shown[0].axes[0])
    assert texts[:4] == ["Data", "Model (y=mx+b)", "$\\mathrm{m} = 2.00$", "$\\mathrm{b} = 0.0100$"]
    assert len(texts) == 5


@pytest.mark.parametrize("b_kwargs, dof", [({}, 2), ({"b": 0}, 2), ({"b": 0.01}, 2)])
def test_chi2_always_divides_by_n_minus_2(shown, b_kwargs, dof):
    p.plot_data(DxVec, FVec, dFVec, m=2, chi2=True, **b_kwargs)
    expected = chi2_by_hand(DxVec, FVec, dFVec, 2, b_kwargs.get("b", 0), len(DxVec) - dof)
    assert legend_chi2(shown[0].axes[0])[-1] == p._chi2_legend_text(expected)


def test_chi2_with_residuals_and_plot_residuals(shown):
    p.plot_data(DxVec, FVec, dFVec, m=2.1, chi2=True, residuals=True)
    p.plot_residuals(DxVec, FVec, dFVec, m=2.1, chi2=True)
    a, r = shown
    assert legend_chi2(a.axes[0])[-1] == legend_chi2(r.axes[0])[-1]


def test_chi2_needs_a_line(shown, capsys):
    with pytest.raises(p.Phys119Error):
        p.plot_data(DxVec, FVec, dFVec, chi2=True)
    assert capsys.readouterr().err.startswith(
        "ERROR: plot_data() needs a line to work out chi-squared for")


@pytest.mark.parametrize("b_kwargs", [{}, {"b": 0.1}])
def test_chi2_needs_at_least_3_points(shown, capsys, b_kwargs):
    with pytest.raises(p.Phys119Error):
        p.plot_data([1.0, 2.0], [2.0, 4.1], [0.1, 0.1], m=2, chi2=True, **b_kwargs)
    assert "needs at least 3 data points to work out chi-squared, got 2" in capsys.readouterr().err
    assert shown == []


def test_autofit_chi2_in_legend_and_result(shown):
    fit = p.autofit(DxVec, FVec, dFVec, chi2=True)
    expected = chi2_by_hand(DxVec, FVec, dFVec, fit.m, fit.b, len(DxVec) - 2)
    assert fit.chi2 == pytest.approx(expected, rel=1e-12)
    assert legend_chi2(shown[0].axes[0])[-1] == p._chi2_legend_text(expected)
    mantissa, _ = p._three_sig_figs(expected)
    assert repr(fit).endswith(f", \u03c7\u00b2 = {mantissa}")


def test_autofit_chi2_available_without_the_legend_line(shown):
    fit = p.autofit(DxVec, FVec, dFVec)
    assert fit.chi2 is not None
    assert len(legend_texts(shown[0].axes[0])) == 2
    assert "\u03c7" not in repr(fit)
    m, b = fit                                     # still unpacks as a pair


def test_autofit_result_with_chi2_survives_copy_and_pickle(shown):
    import copy
    import pickle
    fit = p.autofit(DxVec, FVec, dFVec, chi2=True)
    for clone in (copy.deepcopy(fit), pickle.loads(pickle.dumps(fit))):
        assert clone.chi2 == fit.chi2 and repr(clone) == repr(fit)



# --- m and b in plot_data's legend (format 2, 2026-10-06) ---------------------

@pytest.mark.parametrize("b_kwargs", [{}, {"b": 0}])
def test_b_left_out_or_zero_is_still_a_two_parameter_line(shown, b_kwargs):
    # 1-parameter fits are not used in the course: b left out means b = 0
    p.plot_data(DxVec, FVec, dFVec, m=2.1, **b_kwargs)
    assert legend_texts(shown[0].axes[0]) == ["Data", "Model (y=mx+b)", "$\\mathrm{m} = 2.10$",
                                              "$\\mathrm{b} = 0.00$"]


def test_no_one_parameter_wording_remains():
    import inspect as _inspect
    source = _inspect.getsource(p)
    assert "(no intercept)" not in source and "y = mx (model)" not in source


def test_values_are_three_significant_figures(shown):
    p.plot_data(DxVec, FVec, dFVec, m=2079.66, b=-0.0021)
    texts = legend_texts(shown[0].axes[0])
    assert texts[2:] == ["$\\mathrm{m} = 2.08 \\times 10^{3}$", "$\\mathrm{b} = -0.00210$"]


def test_data_only_legend(shown):
    p.plot_data(DxVec, FVec, dFVec)
    assert legend_texts(shown[0].axes[0]) == ["Data"]


def test_students_model_label_is_kept(shown):
    p.plot_data(DxVec, FVec, dFVec, m=2, b=0.01, model_label="Hooke's law")
    assert legend_texts(shown[0].axes[0])[1] == "Hooke's law"


def test_plot_residuals_has_the_same_legend(shown):
    p.plot_data(DxVec, FVec, dFVec, m=2, residuals=True, chi2=True)
    p.plot_residuals(DxVec, FVec, dFVec, m=2, chi2=True)
    assert legend_texts(shown[0].axes[0]) == legend_texts(shown[1].axes[0])


def test_autofit_legend_is_unchanged(shown):
    fit = p.autofit(DxVec, FVec, dFVec, chi2=True)
    texts = legend_texts(shown[0].axes[0])
    assert texts[0] == f"y = mx + b (best fit)\nm = {fit.m:.4g}\nb = {fit.b:.4g}"
    assert texts[1] == "Data" and texts[2].startswith("$\\chi^2 = ") and len(texts) == 3


# --- The light grid, with and without residuals (2026-10-06) ---------------------

def grid_on(ax):
    lines = ax.xaxis.get_gridlines() + ax.yaxis.get_gridlines()
    return bool(lines) and all(l.get_visible() and l.get_alpha() == 0.3 for l in lines)


@pytest.mark.parametrize("call", [
    lambda: p.plot_data(DxVec, FVec, dFVec),
    lambda: p.plot_data(DxVec, FVec, dFVec, m=2),
    lambda: p.plot_data(DxVec, FVec, dFVec, m=2, residuals=True),
    lambda: p.plot_residuals(DxVec, FVec, dFVec, m=2),
    lambda: p.autofit(DxVec, FVec, dFVec, residuals=False),
])
def test_every_panel_has_the_light_grid(shown, call):
    call()
    assert all(grid_on(ax) for ax in shown[0].axes)



# --- Default legend wording for plot_data and plot_residuals (2026-10-06) -------

@pytest.mark.parametrize("call", [
    lambda: p.plot_data(DxVec, FVec, dFVec, m=2, b=0.01),
    lambda: p.plot_data(DxVec, FVec, dFVec, m=2, b=0.01, residuals=True),
    lambda: p.plot_residuals(DxVec, FVec, dFVec, m=2, b=0.01),
])
def test_default_legend_wording(shown, call):
    call()
    assert legend_texts(shown[0].axes[0])[:2] == ["Data", "Model (y=mx+b)"]


def test_keyword_arguments_replace_the_default_wording(shown):
    p.plot_residuals(DxVec, FVec, dFVec, m=2, data_label="Round 1 data",
                     model_label="Hooke's law, F = k x")
    assert legend_texts(shown[0].axes[0])[:2] == ["Round 1 data", "Hooke's law, F = k x"]


def test_m_and_b_are_upright_like_the_model_line(shown):
    # math text italicises letters; the model entry ("Model (y=mx+b)") is plain text
    p.plot_data(DxVec, FVec, dFVec, m=2, b=0.01)
    texts = legend_texts(shown[0].axes[0])
    assert texts[2] == "$\\mathrm{m} = 2.00$" and texts[3] == "$\\mathrm{b} = 0.0100$"



# --- LaTeX in labels (2026-10-08) ---------------------------------------------

@pytest.mark.parametrize("text", [
    "Force $F$ (N)", "$\\Delta x$ (m)", "Force $F$\n(N, $\\Delta x$)", "Cost \\$5",
    r"$\theta$ (rad)", "$\\frac{1}{2} m v^2$", "Plain title", "Line 1\nLine 2", "$\\nu$ (Hz)",
])
def test_latex_labels_are_drawn(shown, capsys, text):
    p.plot_data(DxVec, FVec, dFVec, m=2, title=text, x_label=text, y_label=text,
                data_label=text, model_label=text, residuals=True)
    ax = shown[0].axes[0]
    assert ax.get_title() == text and legend_texts(ax)[:2] == [text, text]
    shown[0].canvas.draw()
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("text, problem", [
    ("$\\Deltax$ (m)", "problem: Unknown symbol: \\Deltax"),
    ("Force $F (N)", "problem: a $ has no partner"),
    ("$\theta$ (rad)", 'problem: "\\theta" was read as a special character'),
    ("$\frac{1}{2}$", 'problem: "\\frac" was read as a special character'),
    ("$\nu$ (Hz)", 'problem: "\\nu" was read as a special character'),
    ("Force $F\n(N)$", "problem: a line break splits a $...$ pair"),
    ("$\\frac{1}{2$", "problem: a { has no partner }"),
])
@pytest.mark.parametrize("name", ["title", "x_label", "y_label", "data_label", "model_label"])
def test_latex_mistakes_get_a_short_message(shown, capsys, text, problem, name):
    with pytest.raises(p.Phys119Error):
        p.plot_data(DxVec, FVec, dFVec, m=2, **{name: text})
    err = capsys.readouterr().err
    assert err.startswith(f"ERROR: plot_data() could not draw the LaTeX in {name}.")
    assert problem in err
    if "special character" in problem:
        # correct apart from a missing r: both fixes are offered (2026-10-09)
        assert "Either put r before the opening quotation mark:" in err
        assert "or double each backslash:" in err
    else:
        assert "every $ has a partner" in err and "double backslash" in err
    assert shown == []


def test_latex_message_shows_the_label_as_typed(shown, capsys):
    with pytest.raises(p.Phys119Error):
        p.plot_data(DxVec, FVec, dFVec, title="$\frac{1}{2}$")
    # read from the cell, so it is exactly what the student typed
    assert 'got title="$\\frac{1}{2}$"' in capsys.readouterr().err


@pytest.mark.parametrize("fn", ["plot_residuals", "autofit"])
def test_latex_checked_in_other_plotting_functions(shown, capsys, fn):
    kwargs = {"m": 2} if fn == "plot_residuals" else {}
    with pytest.raises(p.Phys119Error):
        getattr(p, fn)(DxVec, FVec, dFVec, y_label="$\\Deltay$", **kwargs)
    assert capsys.readouterr().err.startswith(f"ERROR: {fn}() could not draw the LaTeX in y_label.")


def test_message_box_is_not_typeset_as_maths(monkeypatch):
    # JupyterLab runs MathJax over HTML output; without these classes the
    # "$F$" in the LaTeX hint is typeset as maths and the message is garbled.
    shown_html = []
    monkeypatch.setattr(p, "_notebook_display",
                        lambda: (lambda obj: shown_html.append(obj), lambda s: s))
    with pytest.raises(p.Phys119Error):
        p.plot_data(DxVec, FVec, dFVec, y_label="Force ($N)")
    assert shown_html[0].startswith('<div class="mathjax_ignore tex2jax_ignore"')
    assert "every $ has a partner" in shown_html[0]


# --- Re-reading the student's cell (2026-10-09) -----------------------------

def test_rereading_the_cell_repeats_no_python_warnings(shown, capsys):
    import warnings
    with warnings.catch_warnings():
        # Python's warning about the student's own code, which re-reading
        # their cell must not repeat. (Other libraries' warnings, such as
        # matplotlib's about pyparsing, are not what this is about.)
        warnings.simplefilter("error", SyntaxWarning)
        warnings.filterwarnings("error", message="invalid escape sequence",
                                category=DeprecationWarning)
        with pytest.raises(p.Phys119Error):
            p.plot_data(DxVec, FVec, dFVec, title="$\\Deltax$")
        assert p._quiet_parse('x = "$\\Delta$"') is not None


# --- Raw-string feedback in plot_data (2026-10-09) ----------------------------

def _label_error(capsys):
    return capsys.readouterr().err


def test_raw_string_with_doubled_backslashes(shown, capsys):
    with pytest.raises(p.Phys119Error):
        p.plot_data(DxVec, FVec, dFVec, x_label=r"Displacement $\\Delta x$ (m)")
    err = _label_error(capsys)
    assert 'got x_label=r"Displacement $\\\\Delta x$ (m)"' in err
    assert 'problem: \\\\Delta has two backslashes, but this is a raw string (r"...")' in err
    assert 'In a raw string, each LaTeX command has a single backslash:\n' \
           '  x_label=r"Displacement $\\Delta x$ (m)"' in err


def test_newline_typed_inside_a_raw_string(shown, capsys):
    with pytest.raises(p.Phys119Error):
        p.plot_data(DxVec, FVec, dFVec, x_label=r"Displacement,\n$\Delta x$ (m)")
    err = _label_error(capsys)
    assert err.startswith("ERROR: plot_data() would show \\n as text in x_label, not start a new line.")
    assert 'x_label=r"Displacement," "\\n" r"$\\Delta x$ (m)"' in err


def test_doubled_newline_in_an_ordinary_string(shown, capsys):
    with pytest.raises(p.Phys119Error):
        p.plot_data(DxVec, FVec, dFVec, x_label="Displacement,\\n$\\Delta x$ (m)")
    err = _label_error(capsys)
    assert "For a new line, write \\n with a single backslash:\n" \
           '  x_label="Displacement,\\n$\\\\Delta x$ (m)"' in err


@pytest.mark.parametrize("typed", ["raw", "ordinary"])
def test_latex_outside_dollar_signs(shown, capsys, typed):
    with pytest.raises(p.Phys119Error):
        if typed == "raw":
            p.plot_data(DxVec, FVec, dFVec, x_label=r"Displacement \Delta x (m)")
        else:
            p.plot_data(DxVec, FVec, dFVec, x_label="Displacement \\Delta x (m)")
    err = _label_error(capsys)
    assert err.startswith("ERROR: plot_data() would show \\Delta as text in x_label.")
    fix = ('x_label=r"Displacement $\\Delta x$ (m)"' if typed == "raw"
           else 'x_label="Displacement $\\\\Delta x$ (m)"')
    assert "LaTeX commands only work between a pair of $ signs:\n  " + fix in err


def test_reminder_matches_a_raw_string(shown, capsys):
    with pytest.raises(p.Phys119Error):
        p.plot_data(DxVec, FVec, dFVec, x_label=r"$\Deltax$ (m)")
    err = _label_error(capsys)
    assert 'got x_label=r"$\\Deltax$ (m)"' in err
    assert 'single backslash in a raw string:  r"$\\Delta x$"' in err
    assert "double backslash" not in err


def test_reminder_for_mixed_pieces(shown, capsys):
    with pytest.raises(p.Phys119Error):
        p.plot_data(DxVec, FVec, dFVec, x_label=r"Force $F" "\n" "(N)")
    assert 'single backslash in r"..." pieces' in _label_error(capsys)


@pytest.mark.parametrize("typed, raw_fix, ordinary_fix", [
    ("$\theta$ (rad)", 'x_label=r"$\\theta$ (rad)"', 'x_label="$\\\\theta$ (rad)"'),
    ("Angle,\n$\theta$ (rad)", 'x_label=r"Angle," "\\n" r"$\\theta$ (rad)"',
     'x_label="Angle,\\n$\\\\theta$ (rad)"'),
    ("$\nu$ (Hz)", 'x_label=r"$\\nu$ (Hz)"', 'x_label="$\\\\nu$ (Hz)"'),
])
def test_missing_r_offers_both_fixes(shown, capsys, typed, raw_fix, ordinary_fix):
    with pytest.raises(p.Phys119Error):
        p.plot_data(DxVec, FVec, dFVec, x_label=typed)
    err = _label_error(capsys)
    assert ("Either put r before the opening quotation mark:\n  " + raw_fix +
            "\nor double each backslash:\n  " + ordinary_fix) in err


def test_label_in_a_variable_gets_a_general_fix(shown, capsys):
    label = r"Displacement,\n$\Delta x$ (m)"
    with pytest.raises(p.Phys119Error):
        p.plot_data(DxVec, FVec, dFVec, x_label=label)
    err = _label_error(capsys)
    assert "got 'Displacement,\\\\n$\\\\Delta x$ (m)'" in err
    assert "in an ordinary\nstring (not a raw one):" in err


def test_autofit_positional_labels_are_read(shown, capsys):
    with pytest.raises(p.Phys119Error):
        p.autofit(DxVec, FVec, dFVec, r"Title", r"$\\Delta x$")
    assert 'got x_label=r"$\\\\Delta x$"' in _label_error(capsys)


@pytest.mark.parametrize("label", [
    r"Displacement," "\n" r"$\Delta x$ (m)", "Displacement,\n$\\Delta x$ (m)",
    r"Frequency $\nu$ (Hz)", "Cost \\$5", "Plain label", r"$\frac{1}{2} m v^2$",
])
def test_correct_labels_still_draw(shown, label):
    p.plot_data(DxVec, FVec, dFVec, x_label=label)
    assert shown[0].axes[0].get_xlabel() == label
