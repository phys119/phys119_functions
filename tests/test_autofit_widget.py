"""Tests for autofit's widget: leaving points out of the fit and putting them
back (dev copy 1.5.0.dev0, where autofit took over from autofit_2).

The choice of points reaches Python as the widget's `excluded` list, as a
click or a tick in the table sends it; these tests set it directly. The
widget's JavaScript (clicks, table, hover) is checked in real JupyterLab by
claude-scratch/scripts/ui_autofit_browser_check.py.

Run with:  python -m pytest tests
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phys119_functions as p  # noqa: E402

anywidget = pytest.importorskip("anywidget")

# Point 4 is about 2 N too high.
X = np.array([1.0, 2, 3, 4, 5, 6, 7, 8])
Y = np.array([2.1, 3.9, 6.2, 11.5, 9.8, 12.1, 14.2, 15.8])
DY = np.array([0.3] * 8)


@pytest.fixture
def drawn(monkeypatch):
    """Each figure autofit draws, newest last, with its two graphs."""
    figures = []
    draw = p._autofit_draw

    def recorded(*args, **kwargs):
        result = draw(*args, **kwargs)
        figures.append(result)
        return result

    monkeypatch.setattr(p, "_autofit_draw", recorded)
    monkeypatch.setattr(p, "_autofits", {})
    yield figures
    plt.close("all")


def last_fit():
    return list(p._autofits.values())[-1]


def legend_texts(ax):
    return [t.get_text() for t in ax.get_legend().get_texts()]


def fit_by_hand(keep):
    m, b = p._weighted_line_fit(X[keep], Y[keep], DY[keep])
    dm, db = p._line_fit_uncertainties(X[keep], DY[keep])
    chi2 = np.sum(((Y[keep] - (m * X[keep] + b)) / DY[keep])**2) / (np.sum(keep) - 2)
    return m, dm, b, db, chi2


# --- leaving points out ----------------------------------------------------------

def test_every_point_is_in_the_fit_at_first(drawn):
    p.autofit(X, Y, DY, pick=True)
    fit = last_fit()
    assert fit.excluded == []
    assert (fit.m, fit.dm, fit.b, fit.db, fit.chi2) == pytest.approx(
        fit_by_hand(np.ones(8, bool)), rel=1e-12)


def test_leaving_a_point_out_refits_without_it(drawn):
    p.autofit(X, Y, DY, pick=True)
    fit = last_fit()
    fit._widget.excluded = [4]
    keep = np.arange(1, 9) != 4
    assert fit.excluded == [4]
    assert (fit.m, fit.dm, fit.b, fit.db, fit.chi2) == pytest.approx(fit_by_hand(keep), rel=1e-12)
    assert fit._widget.message == ""
    m, b = fit                                     # still unpacks as a pair
    assert (m, b) == (fit.m, fit.b)


def test_putting_it_back_restores_the_first_fit(drawn):
    p.autofit(X, Y, DY, pick=True)
    fit = last_fit()
    first = (fit.m, fit.b, fit.chi2)
    fit._widget.excluded = [4]
    fit._widget.excluded = []
    assert (fit.m, fit.b, fit.chi2) == first


def test_points_left_out_are_drawn_hollow_and_numbered(drawn):
    p.autofit(X, Y, DY, pick=True)
    last_fit()._widget.excluded = [4]
    fig, top, bottom, _ = drawn[-1]
    texts = legend_texts(top)
    assert texts[:3] == ["Data", "Excluded from fit", "Best fit (y=mx+b)"]
    fit = last_fit()
    assert texts[3:5] == [p._uncertainty_legend_text("m", fit.m, fit.dm),
                          p._uncertainty_legend_text("b", fit.b, fit.db)]
    numbers = [t.get_text() for t in top.texts if t.get_visible()]
    assert numbers == ["4"]
    # the filled points are the 7 in the fit
    assert len(top.containers[0].lines[0].get_xdata()) == 7


def test_the_widget_tells_the_browser_each_time(drawn):
    p.autofit(X, Y, DY, pick=True)
    widget = last_fit()._widget
    before, png = widget.revision, widget.png
    widget.excluded = [4]
    assert widget.revision == before + 1 and widget.png != png


@pytest.mark.parametrize("choice, says", [
    ([1, 2, 3, 4, 5, 6], "at least 3 points must stay in the fit"),
])
def test_too_few_points_left_is_refused(drawn, choice, says):
    p.autofit(X, Y, DY, pick=True)
    fit = last_fit()
    fit._widget.excluded = [4]
    values = (fit.m, fit.b)
    fit._widget.excluded = choice
    assert says in fit._widget.message
    assert fit._widget.excluded == [4] and fit.excluded == [4]     # undone
    assert (fit.m, fit.b) == values


def test_one_x_value_left_is_refused(drawn):
    xs = np.array([1.0, 1, 1, 2, 3])
    p.autofit(xs, [2.0, 2.1, 1.9, 4.0, 6.1], [0.1] * 5, pick=True)
    fit = last_fit()
    fit._widget.excluded = [4, 5]
    assert "at least two different x values" in fit._widget.message
    assert fit.excluded == []


def test_a_failed_redraw_is_undone_with_a_message(drawn, monkeypatch):
    p.autofit(X, Y, DY, pick=True)
    fit = last_fit()
    monkeypatch.setattr(p, "_autofit_figure", lambda *a, **k: 1 / 0)
    fit._widget.excluded = [2, 4]
    assert fit._widget.excluded == [] and fit.excluded == []
    assert fit._widget.message.startswith("Something went wrong redrawing the fit.")


# --- the residuals graph once points are left out ------------------------------

def test_residuals_are_scaled_to_the_points_in_the_fit(drawn):
    p.autofit(X, Y, DY, pick=True)
    fit = last_fit()
    fit._widget.excluded = [4]
    _, _, bottom, resid_shown = drawn[-1]
    keep = np.arange(1, 9) != 4
    resid = Y - (fit.m * X + fit.b)
    lo, hi = bottom.get_ylim()
    assert lo <= min(0, np.min(resid[keep] - DY[keep]))
    assert hi >= max(0, np.max(resid[keep] + DY[keep]))
    # point 4 is far above: it is shown as an arrow at the top edge
    assert resid[3] > hi and lo < resid_shown[3] < hi
    assert [t.get_text() for t in bottom.texts if t.get_text()] == ["4"]


def test_with_every_point_in_the_residuals_are_scaled_as_plot_data_does(drawn):
    p.autofit(X, Y, DY, pick=True)
    _, _, bottom, _ = drawn[-1]
    resid = Y - (last_fit().m * X + last_fit().b)
    lo, hi = bottom.get_ylim()
    assert lo < np.min(resid - DY) and hi > np.max(resid + DY)


def test_the_arrow_can_be_clicked_where_it_is_drawn(drawn):
    p.autofit(X, Y, DY, pick=True)
    widget = last_fit()._widget
    widget.excluded = [4]
    where = widget.positions["resid"][3]
    assert 0 < where[0] < widget.positions["width"]
    assert 0 < where[1] < widget.positions["height"]
    # above every residual in the fit
    assert all(where[1] < q[1] for i, q in enumerate(widget.positions["resid"]) if i != 3)


def test_the_picture_does_not_shift_sideways(drawn):
    p.autofit(X, Y, DY, pick=True)
    widget = last_fit()._widget
    before = widget.positions
    widget.excluded = [4]
    after = widget.positions
    assert after["left_edge"] <= before["left_edge"]
    for a, b in zip(before["main"], after["main"]):
        assert abs(a[0] - b[0]) < 0.5


@pytest.mark.parametrize("note, expected", [(False, None), (True, "Off-scale points: 1 above")])
def test_note_counts_the_off_scale_points(drawn, note, expected):
    p.autofit(X, Y, DY, note=note, pick=True)
    last_fit()._widget.excluded = [4]
    _, _, bottom, _ = drawn[-1]
    notes = [t.get_text() for t in bottom.texts if t.get_text().startswith("Off-scale")]
    assert notes == ([] if expected is None else [expected])


def test_no_note_while_nothing_is_off_scale(drawn):
    p.autofit(X, Y, DY, note=True, pick=True)
    last_fit()._widget.excluded = [5]              # a point near the line
    _, _, bottom, _ = drawn[-1]
    assert not any(t.get_text().startswith("Off-scale") for t in bottom.texts)


def test_crowded_numbers_never_overlap(drawn):
    # 40 points, the first 30 left out and far off the line
    xs = np.arange(1.0, 41)
    ys = 2 * xs + np.where(xs <= 30, 50.0, 0.0)
    p.autofit(xs, ys, np.ones(40), pick=True)
    last_fit()._widget.excluded = list(range(1, 31))
    fig, top, bottom, _ = drawn[-1]
    renderer = fig.canvas.get_renderer()
    for ax in (top, bottom):
        boxes = [t.get_window_extent(renderer) for t in ax.texts
                 if t.get_visible() and t.get_text()]
        assert boxes                                  # some numbers are shown
        assert len(boxes) < 30                        # and some left off
        for i, a in enumerate(boxes):
            assert not any(a.overlaps(b) for b in boxes[i + 1:])


# --- remembered, and the keyword rules -------------------------------------------

@pytest.fixture
def notebook(tmp_path, monkeypatch):
    """autofit run as cells of a saved notebook: run("A", "note=True") runs
    the cell `autofit(X, Y, DY, note=True, pick=True)` with cell id A
    (pick=False leaves pick=True out). The cell's source
    is registered as IPython registers a cell's, so the call as typed can be
    read back."""
    import linecache
    monkeypatch.setenv("JPY_SESSION_NAME", str(tmp_path / "Prelab10.ipynb"))
    monkeypatch.chdir(tmp_path)
    cell = {"id": None, "count": 0}
    monkeypatch.setattr(p, "_current_cell_id", lambda: cell["id"])
    namespace = {"autofit": p.autofit, "X": X, "Y": Y, "DY": DY}

    def run(cell_id, keywords="", pick=True, **data):
        namespace.update(data)                      # e.g. Y=... edits the data
        cell["id"] = cell_id
        cell["count"] += 1
        keywords = ", ".join(k for k in (keywords, "pick=True" if pick else "") if k)
        source = "autofit(X, Y, DY" + (", " + keywords if keywords else "") + ")\n"
        filename = f"<cell {cell['count']}>"
        linecache.cache[filename] = (len(source), None, [source], filename)
        exec(compile(source, filename, "exec"), namespace)
        fit = last_fit()
        assert namespace[fit.name] is fit           # the name is a notebook variable
        return fit

    run.store = tmp_path / ".Prelab10.manual_fits.json"
    run.saved = lambda: json.loads(run.store.read_text())["fits"]
    run.restart = lambda: monkeypatch.setattr(p, "_autofits", {})
    return run


def test_points_left_out_are_remembered_after_a_restart(drawn, notebook):
    fit = notebook("A")
    fit._widget.excluded = [4]
    [entry] = notebook.saved().values()
    assert entry["name"] == "autofit1" and entry["excluded"] == [4]
    notebook.restart()
    again = notebook("A")
    assert again.name == "autofit1" and again.excluded == [4]
    assert again.m == pytest.approx(fit.m, rel=1e-12)


@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("edit", ["a point added at the start", "one value changed"])
def test_points_left_out_are_forgotten_when_the_data_change(drawn, notebook, restart, edit):
    # Points are left out by number: after an edit, the same number may be
    # a different point, so every point goes back in. The name stays.
    notebook("A")._widget.excluded = [4]
    if restart:
        notebook.restart()
    if edit == "one value changed":
        data = {"Y": np.where(np.arange(len(Y)) == 0, Y[0] + 0.1, Y)}
    else:
        data = {"X": np.r_[0.0, X], "Y": np.r_[0.2, Y], "DY": np.r_[DY[0], DY]}
    fit = notebook("A", **data)
    assert fit.name == "autofit1" and fit.excluded == []
    assert fit._widget.excluded == []
    # and the new choice is remembered for the new data
    fit._widget.excluded = [5]
    notebook.restart()
    assert notebook("A").excluded == [5]


def test_changing_note_keeps_the_name_and_the_choice(drawn, notebook):
    notebook("A")._widget.excluded = [4]
    fit = notebook("A", "note=True")
    assert fit.name == "autofit1" and fit.excluded == [4]
    assert len(notebook.saved()) == 1


def test_another_call_is_another_fit(drawn, notebook):
    notebook("A")._widget.excluded = [4]
    fit = notebook("A", "chi2=False")
    assert fit.name == "autofit2" and fit.excluded == []


@pytest.mark.parametrize("saved", [[9], [0], [True], ["4"], [1, 2, 3, 4, 5, 6], "4"])
def test_a_saved_choice_that_cannot_be_used_is_ignored(drawn, notebook, saved):
    notebook("A")
    data = json.loads(notebook.store.read_text())
    key = next(iter(data["fits"]))
    data["fits"][key]["excluded"] = saved
    notebook.store.write_text(json.dumps(data))
    notebook.restart()
    fit = notebook("A")
    assert fit.name == "autofit1" and fit.excluded == []


@pytest.mark.parametrize("bad", ["yes", 1, None])
@pytest.mark.parametrize("keyword", ["note", "pick"])
def test_note_and_pick_must_be_true_or_false(drawn, capsys, bad, keyword):
    with pytest.raises(p.Phys119Error):
        p.autofit(X, Y, DY, **{keyword: bad})
    assert capsys.readouterr().err.startswith(
        f"ERROR: autofit() needs {keyword} to be True or False.")


@pytest.mark.parametrize("keyword", ["exclude", "select", "off_scale", "summary"])
def test_draft_keywords_are_gone(drawn, capsys, keyword):
    with pytest.raises(p.Phys119Error):
        p.autofit(X, Y, DY, **{keyword: None})
    assert f"unexpected keyword argument '{keyword}'" in capsys.readouterr().err


def test_autofit_2_is_gone():
    assert not hasattr(p, "autofit_2") and "autofit_2" not in p.__all__


# --- in a notebook ---------------------------------------------------------------

class FakeHandle:
    def __init__(self):
        self.updates = []

    def update(self, bundle, raw=False):
        self.updates.append(bundle)


def test_in_a_notebook_widget_picture_and_a_summary_that_follows(drawn, monkeypatch):
    shown = []
    handle = FakeHandle()

    def display(bundle, raw=False, metadata=None, display_id=None):
        shown.append((bundle, metadata))
        return handle if display_id else None

    monkeypatch.setattr(p, "_notebook_display", lambda: (display, None))
    p.autofit(X, Y, DY, pick=True)
    fit = last_fit()
    (figure, metadata), (summary, _) = shown
    # one output: the widget, with a still picture for anything that cannot show it
    assert "application/vnd.jupyter.widget-view+json" in figure
    assert figure["image/png"][:8] == b"\x89PNG\r\n\x1a\n"
    assert metadata["image/png"]["width"] == round(fit._widget.positions["width"])
    assert summary["text/plain"].startswith(repr(fit) + "\n")
    fit._widget.excluded = [4]
    assert handle.updates[-1]["text/plain"].startswith(repr(fit) + "\n")


def test_outside_a_notebook_the_summary_is_printed(drawn, capsys):
    p.autofit(X, Y, DY, pick=True)
    fit = last_fit()
    n = fit.name
    assert capsys.readouterr().out == (
        f"{repr(fit)}\n{n}.m, {n}.dm, {n}.b, {n}.db, {n}.chi2 and {n}.excluded hold these "
        "values at full precision.\n")


# --- pick=False, the default: every point in, no widget ---------------------------

def test_by_default_every_point_is_fitted_and_there_is_no_widget(drawn):
    p.autofit(X, Y, DY)
    fit = last_fit()
    assert fit._widget is None and fit.excluded == []
    assert (fit.m, fit.dm, fit.b, fit.db, fit.chi2) == pytest.approx(
        fit_by_hand(np.ones(8, bool)), rel=1e-12)
    _, top, _, _ = drawn[-1]
    assert legend_texts(top)[:2] == ["Data", "Best fit (y=mx+b)"]


def test_by_default_a_notebook_shows_only_the_picture_and_the_summary(drawn, monkeypatch):
    shown = []

    def display(bundle, raw=False, metadata=None, display_id=None):
        shown.append((bundle, metadata))

    monkeypatch.setattr(p, "_notebook_display", lambda: (display, None))
    p.autofit(X, Y, DY)
    fit = last_fit()
    (figure, metadata), (summary, _) = shown
    assert set(figure) == {"image/png", "text/plain"}         # no widget
    assert figure["image/png"][:8] == b"\x89PNG\r\n\x1a\n"
    assert metadata["image/png"]["width"] > 0
    n = fit.name
    assert summary["text/plain"] == (
        f"{repr(fit)}\n{n}.m, {n}.dm, {n}.b, {n}.db and {n}.chi2 hold these values "
        "at full precision.")


def test_by_default_the_summary_does_not_mention_excluded(drawn, capsys):
    p.autofit(X, Y, DY)
    out = capsys.readouterr().out
    assert "excluded" not in out and "chi2 hold these values" in out


def test_turning_pick_off_and_on_keeps_the_name_and_the_choice(drawn, notebook):
    notebook("A")._widget.excluded = [4]
    off = notebook("A", pick=False)
    assert off.name == "autofit1" and off.excluded == [] and off._widget is None
    assert off.m == pytest.approx(fit_by_hand(np.ones(8, bool))[0], rel=1e-12)
    [entry] = notebook.saved().values()
    assert entry["excluded"] == [4]                  # kept for when pick comes back
    notebook.restart()
    on = notebook("A")
    assert on.name == "autofit1" and on.excluded == [4]


def test_a_first_run_without_pick_keeps_its_name(drawn, notebook):
    assert notebook("A", pick=False).name == "autofit1"
    notebook.restart()
    assert notebook("A", pick=False).name == "autofit1"
    assert notebook("B", pick=False).name == "autofit2"
