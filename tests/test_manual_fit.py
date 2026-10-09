"""Tests for manual_fit, the interactive fit drawn in the notebook.

Layers:
  - the Python side (checks, ranges), which runs anywhere;
  - the widget object, its naming and its memory, which need anywidget and
    are skipped without it;
  - the widget's JavaScript, run in headless Microsoft Edge or Chrome with a
    stand-in for the notebook's widget model, skipped if neither browser is
    found. It checks behaviour (sliders, typed values, Reset, keys kept from
    JupyterLab), not appearance; appearance is checked by eye in JupyterLab.

Run with:  python -m pytest tests
"""

import html
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phys119_functions as p  # noqa: E402


# Prelab05_new.ipynb cell 23: spring data
DxVec = np.array([0.00, 0.12, 0.21, 0.29, 0.40, 0.48, 0.60])
FVec = np.array([0.00, 0.26, 0.43, 0.63, 0.81, 1.05, 1.23])
dFVec = np.array([0.02, 0.02, 0.03, 0.03, 0.03, 0.04, 0.04])

# Prelab05_new.ipynb cell 63: Ohm's law data
IVec = np.array([0.000, 0.009, 0.020, 0.029, 0.043, 0.053, 0.070, 0.078])
Vr1Vec = np.array([0.00, 0.11, 0.19, 0.28, 0.41, 0.53, 0.67, 0.80])
dVr1Vec = np.array([0.02, 0.03, 0.02, 0.02, 0.03, 0.02, 0.03, 0.03])

try:
    import anywidget  # noqa: F401
    HAVE_ANYWIDGET = True
except ImportError:
    HAVE_ANYWIDGET = False
needs_anywidget = pytest.mark.skipif(not HAVE_ANYWIDGET, reason="anywidget not installed")


# --- Python side ------------------------------------------------------------

def test_exported():
    assert "manual_fit" in p.__all__


@pytest.mark.parametrize("data", [(DxVec, FVec, dFVec), (IVec, Vr1Vec, dVr1Vec)])
def test_automatic_ranges_contain_the_best_fit_and_give_nothing_away(data):
    x, y, dy = data
    (m_lo, m_hi, m_step), (b_lo, b_hi, b_step) = p._manual_fit_ranges(x, y, dy)
    m_best, b_best = np.polyfit(x, y, 1, w=1 / dy)
    assert m_lo == -m_hi                       # centred on zero, not on the answer
    assert m_lo < m_best < m_hi
    assert b_lo < b_best < b_hi
    assert 500 <= (m_hi - m_lo) / m_step <= 2500
    assert 500 <= (b_hi - b_lo) / b_step <= 2500


def test_range_values_carry_no_float_noise():
    ranges = p._manual_fit_ranges(IVec, Vr1Vec, dVr1Vec)
    for value in ranges[0] + ranges[1]:
        assert len(repr(value)) < 12


def test_keywords_set_the_ranges_and_the_step_follows_the_data():
    # one step moves the furthest residual by about 1/10 of the smallest
    # error bar (0.02): slope 0.1*0.02/0.6 -> 0.002, intercept 0.002
    m_range, b_range = p._manual_fit_slider_ranges(
        DxVec, FVec, dFVec, 1.5, 2.5, -0.2, 0.2)
    assert m_range == [1.5, 2.5, 0.002]
    assert b_range == [-0.2, 0.2, 0.002]


@pytest.mark.parametrize("data, limits, steps", [
    ((IVec, Vr1Vec, dVr1Vec), (5, 15, -0.2, 0.2), (0.02, 0.002)),     # the V4 Ohm fits
    # Your turn #10: a tenth of an error bar would be 0.002, 1000 positions,
    # so the 500-position limit makes it 0.005
    ((DxVec, FVec, dFVec), (1, 3, -0.5, 0.5), (0.005, 0.002)),
])
def test_slider_steps_are_a_tenth_of_an_error_bar(data, limits, steps):
    m_range, b_range = p._manual_fit_slider_ranges(*data, *limits)
    assert (m_range[2], b_range[2]) == steps
    x, _, dy = data
    effect = m_range[2] * np.max(np.abs(x)) / np.min(dy)
    positions = (m_range[1] - m_range[0]) / m_range[2]
    assert 0.05 <= effect <= 0.1 or round(positions) == 400


def test_slider_positions_stay_between_50_and_500():
    m_range, b_range = p._manual_fit_slider_ranges(DxVec, FVec, dFVec, None, None, None, None)
    for lo, hi, step in (m_range, b_range):
        assert 50 <= round((hi - lo) / step) <= 500
    m_range, _ = p._manual_fit_slider_ranges(DxVec, FVec, dFVec, 2.0, 2.01, None, None)
    assert round((m_range[1] - m_range[0]) / m_range[2]) >= 50


def test_steps_keywords_count_positions_including_both_ends():
    # The case reported on 2026-10-06: 1.5 to 2.5 in 11 positions is 0.1.
    m_range, _ = p._manual_fit_slider_ranges(DxVec, FVec, dFVec, 1.5, 2.5, -0.5, 0.5,
                                             m_steps=11)
    assert m_range == [1.5, 2.5, 0.1]
    m_range, b_range = p._manual_fit_slider_ranges(
        IVec, Vr1Vec, dVr1Vec, 5, 15, -0.2, 0.2, m_steps=101, b_steps=41)
    assert m_range == [5.0, 15.0, 0.1]
    assert b_range == [-0.2, 0.2, 0.01]
    m_range, b_range = p._manual_fit_slider_ranges(
        IVec, Vr1Vec, dVr1Vec, 5, 15, -0.2, 0.2, m_steps=101)
    assert b_range[2] == 0.002                       # the other keeps the rule


@pytest.mark.parametrize("bad", [0, 1, 2.5, "100", True, -5])
def test_steps_keywords_must_be_whole_numbers(capsys, bad):
    with pytest.raises(p.Phys119Error):
        p.manual_fit(DxVec, FVec, dFVec, m_steps=bad)
    assert capsys.readouterr().err.startswith(
        "ERROR: manual_fit() needs m_steps to be a whole number, 2 or more.")


def test_keywords_left_out_come_from_the_data():
    auto_m, auto_b = p._manual_fit_ranges(DxVec, FVec, dFVec)
    m_range, b_range = p._manual_fit_slider_ranges(
        DxVec, FVec, dFVec, 1.5, None, None, None)
    assert m_range[:2] == [1.5, auto_m[1]]
    assert b_range[:2] == auto_b[:2]


@pytest.mark.parametrize("kwargs, expected", [
    ({"m_min": 3, "m_max": 2}, "needs m_min to be below m_max.\n  m_min = 3\n  m_max = 2"),
    ({"b_min": 100}, "needs b_min to be below b_max.\n  b_min = 100\n"
                     "  b_max = 5.14 (chosen from the data)"),
    ({"m_max": "2"}, "needs m_max to be an ordinary number.\n  got str: '2'"),
    ({"b_min": np.nan}, "needs b_min to be an ordinary number."),
])
def test_bad_ranges_are_a_red_box(capsys, kwargs, expected):
    with pytest.raises(p.Phys119Error):
        p.manual_fit(DxVec, FVec, dFVec, **kwargs)
    assert expected in capsys.readouterr().err


@pytest.mark.parametrize("bad, expected", [
    ((DxVec, FVec, 0.03), "needs dy to hold one uncertainty for each data point"),
    ((DxVec, [0, 0.26, np.nan, 0.63, 0.81, 1.05, 1.23], dFVec), "missing values (nan) in y"),
    ((DxVec, FVec, dFVec[:5]), "needs x, y and dy to be the same length"),
    (([0.2] * 4, FVec[:4], dFVec[:4]), "needs at least two different x values"),
])
def test_bad_input_stops_before_any_widget(capsys, bad, expected):
    with pytest.raises(p.Phys119Error):
        p.manual_fit(*bad)
    err = capsys.readouterr().err
    assert err.startswith("ERROR: manual_fit()")
    assert expected in err


def test_missing_anywidget_is_a_red_box(capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "anywidget", None)        # import now fails
    monkeypatch.setattr(p, "_manual_fit_widget_class", None)
    with pytest.raises(p.Phys119Error):
        p.manual_fit(DxVec, FVec, dFVec)
    err = capsys.readouterr().err
    assert err.startswith("ERROR: manual_fit() needs the anywidget package")
    assert "This is not a mistake in your code." in err


# --- The widget object, naming and memory -------------------------------------

def _fit_names():
    return [n for n in globals() if n.startswith("manual_fit") and n[10:].isdigit()]


@pytest.fixture
def notebook(tmp_path, monkeypatch):
    """A fresh kernel in a notebook's folder: no fits yet, no saved file."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JPY_SESSION_NAME", str(tmp_path / "Prelab05.ipynb"))
    monkeypatch.setattr(p, "_manual_fits", {})
    yield tmp_path
    for name in _fit_names():
        del globals()[name]


def saved(folder):
    return json.loads((folder / ".Prelab05.manual_fits.json").read_text())["fits"]


@needs_anywidget
def test_name_and_starting_values_are_printed(notebook, capsys):
    p.manual_fit(DxVec, FVec, dFVec, m_min=1.5, m_max=2.5, b_min=-0.2, b_max=0.4)
    assert capsys.readouterr().out == (
        "manual_fit1.m and manual_fit1.b hold your final slope and intercept "
        "values, wherever you leave the sliders.\n"
        "Line started at: m = 2, b = 0.1\n")


@needs_anywidget
def test_after_a_rerun_the_printed_values_are_where_the_line_was_left(notebook, capsys):
    p.manual_fit(DxVec, FVec, dFVec)
    globals()["manual_fit1"]._widget.m = 2.08
    globals()["manual_fit1"]._widget.b = 0.0047
    capsys.readouterr()
    p.manual_fit(DxVec, FVec, dFVec)
    assert "Line started at: m = 2.08, b = 0.0047\n" in capsys.readouterr().out


@needs_anywidget
def test_in_a_notebook_the_line_is_html_that_exports_keep(notebook, monkeypatch):
    shown = []
    monkeypatch.setattr(p, "_notebook_display",
                        lambda: (lambda obj, **kwargs: shown.append(obj), p_html()))
    p.manual_fit(DxVec, FVec, dFVec, m_min=1.5, m_max=2.5)
    text = shown[0].data
    assert "<code>manual_fit1.m</code> and <code>manual_fit1.b</code> hold your final" in text
    assert "Line started at: m = 2, b = " in text
    assert "--jp-ui-font-color2" in text          # the starting values are quieter


def p_html():
    from IPython.display import HTML
    return HTML


@needs_anywidget
def test_fit_is_named_and_created_as_a_variable(notebook):
    result = p.manual_fit(DxVec, FVec, dFVec)
    assert result is None
    fit = globals()["manual_fit1"]
    assert fit.name == "manual_fit1"
    assert fit._widget.name == "manual_fit1"


@needs_anywidget
def test_line_starts_in_the_middle_of_the_ranges(notebook):
    p.manual_fit(DxVec, FVec, dFVec, m_min=1.5, m_max=2.5, b_min=-0.2, b_max=0.4)
    fit = globals()["manual_fit1"]
    assert (fit.m, fit.b) == (2.0, 0.1)
    assert fit._widget.start == [2.0, 0.1]


@needs_anywidget
def test_values_follow_the_widget(notebook):
    p.manual_fit(list(DxVec), list(FVec), list(dFVec))
    fit = globals()["manual_fit1"]
    fit._widget.m = 2.08          # as if the student moved the slider
    fit._widget.b = 0.0047
    assert (fit.m, fit.b) == (2.08, 0.0047)
    assert type(fit.m) is float
    assert repr(fit) == "manual_fit1: m = 2.08, b = 0.0047"


@needs_anywidget
def test_different_calls_get_different_names(notebook):
    p.manual_fit(DxVec, FVec, dFVec)
    p.manual_fit(IVec, Vr1Vec, dVr1Vec)
    assert globals()["manual_fit1"]._widget.x == DxVec.tolist()
    assert globals()["manual_fit2"]._widget.x == IVec.tolist()


@needs_anywidget
def test_rerunning_a_call_keeps_its_name_and_line(notebook):
    for run in range(2):
        p.manual_fit(DxVec, FVec, dFVec)
        if run == 0:
            globals()["manual_fit1"]._widget.m = 2.08
    assert _fit_names() == ["manual_fit1"]
    assert globals()["manual_fit1"].m == 2.08


@needs_anywidget
def test_layout_of_the_call_does_not_matter(notebook):
    p.manual_fit(DxVec, FVec, dFVec)
    p.manual_fit(
        DxVec,
        FVec,
        dFVec,
    )
    assert _fit_names() == ["manual_fit1"]


@needs_anywidget
def test_line_is_saved_and_restored_after_a_restart(notebook, monkeypatch):
    p.manual_fit(DxVec, FVec, dFVec)
    p.manual_fit(IVec, Vr1Vec, dVr1Vec)
    globals()["manual_fit2"]._widget.m = 9.9
    assert saved(notebook)["call: p.manual_fit(IVec, Vr1Vec, dVr1Vec)"] == {
        "name": "manual_fit2", "m": 9.9, "b": globals()["manual_fit2"].b}

    monkeypatch.setattr(p, "_manual_fits", {})            # a kernel restart
    for name in _fit_names():
        del globals()[name]
    p.manual_fit(IVec, Vr1Vec, dVr1Vec)            # cells run out of order
    assert _fit_names() == ["manual_fit2"]
    assert globals()["manual_fit2"].m == 9.9


@needs_anywidget
def test_reset_target_is_the_middle_even_after_a_restore(notebook, monkeypatch):
    p.manual_fit(DxVec, FVec, dFVec, m_min=1, m_max=3)
    globals()["manual_fit1"]._widget.m = 2.5
    monkeypatch.setattr(p, "_manual_fits", {})
    p.manual_fit(DxVec, FVec, dFVec, m_min=1, m_max=3)
    widget = globals()["manual_fit1"]._widget
    assert widget.m == 2.5 and widget.start[0] == 2.0


@needs_anywidget
def test_a_damaged_memory_file_is_ignored(notebook):
    (notebook / ".Prelab05.manual_fits.json").write_text("{ not json")
    p.manual_fit(DxVec, FVec, dFVec)
    assert globals()["manual_fit1"].m == 0.0
    assert saved(notebook)             # and it is written afresh


@needs_anywidget
def test_no_notebook_means_memory_for_this_kernel_only(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("JPY_SESSION_NAME", raising=False)
    monkeypatch.setattr(p, "_manual_fits", {})
    p.manual_fit(DxVec, FVec, dFVec)
    assert globals().pop("manual_fit1").name == "manual_fit1"
    assert list(tmp_path.iterdir()) == []


@needs_anywidget
def test_axes_show_the_typed_names(notebook):
    p.manual_fit(DxVec, FVec, dFVec)
    labels = globals()["manual_fit1"]._widget.labels
    assert (labels["x_label"], labels["y_label"]) == ("DxVec", "FVec")


# --- The JavaScript, in a headless browser ------------------------------------

def _find_browser():
    for candidate in (
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "/usr/bin/google-chrome", "/usr/bin/chromium", "/usr/bin/chromium-browser",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


BROWSER = _find_browser()
needs_browser = pytest.mark.skipif(BROWSER is None, reason="no headless Edge or Chrome")

_HARNESS = """<!doctype html><html><head><meta charset="utf-8"></head><body>
<div id="widget"></div><pre id="out">not run</pre>
<script type="module">
const log = [];
try {
  const state = __STATE__;
  const listeners = {};
  let saves = 0;
  const model = {
    get: (k) => state[k],
    set: (k, v) => { state[k] = v; (listeners["change:" + k] || []).forEach((f) => f()); },
    save_changes: () => { saves += 1; },
    on: (ev, f) => { (listeners[ev] = listeners[ev] || []).push(f); },
    off: (ev, f) => { listeners[ev] = (listeners[ev] || []).filter((g) => g !== f); },
  };
  // Record every piece of text drawn on the canvases, to read the legend.
  const drawn = [];
  const original = CanvasRenderingContext2D.prototype.fillText;
  CanvasRenderingContext2D.prototype.fillText = function (text, ...rest) {
    drawn.push(String(text)); return original.call(this, text, ...rest); };
  const mod = await import(URL.createObjectURL(
    new Blob([__ESM__], { type: "text/javascript" })));
  const el = document.getElementById("widget");
  const cleanup = mod.default.render({ model, el });
  const [mSlider, bSlider] = el.querySelectorAll("input[type=range]");
  const [mBox, bBox] = el.querySelectorAll("input[type=number]");
  const record = (key, value) => log.push(key + "=" + value);
  record("canvases", el.querySelectorAll("canvas").length);
  record("chi2_start", drawn.filter((t) => t.startsWith("\u03c7\u00b2 = ")).pop());
  record("start_m_box", mBox.value);
  const [mLo, mHi, mStep] = state.m_range;
  record("slider_positions", mSlider.min + " " + mSlider.max + " " + mSlider.step);
  mSlider.value = mSlider.max; mSlider.dispatchEvent(new Event("input"));
  record("at_last_position", state.m === mHi);
  mSlider.value = "0"; mSlider.dispatchEvent(new Event("input"));
  record("at_first_position", state.m === mLo);
  record("start_b_box", bBox.value);
  mSlider.value = String(Math.round((2.05 - mLo) / mStep)); mSlider.dispatchEvent(new Event("input"));
  record("slider_m", state.m); record("slider_m_box", mBox.value);
  mBox.value = "1.234"; mBox.dispatchEvent(new Event("change"));
  record("typed_m", state.m);
  bBox.value = "0.0047"; bBox.dispatchEvent(new Event("change"));
  record("typed_b", state.b);
  model.set("m", 50);
  record("wide_max", parseFloat((mLo + mSlider.max * mStep).toPrecision(12)));
  record("wide_box", mBox.value);
  mBox.value = ""; mBox.dispatchEvent(new Event("change"));
  record("after_blank_m", state.m);
  let leaked = false;
  document.body.addEventListener("keydown", () => { leaked = true; });
  mBox.dispatchEvent(new KeyboardEvent("keydown", { key: "1", bubbles: true }));
  record("key_leaked", leaked);
  // The number boxes' arrows move by one slider step, keeping any offset.
  mBox.focus();
  mBox.value = "2.08"; mBox.dispatchEvent(new Event("change"));
  record("box_step", mBox.step);
  mBox.stepUp(); mBox.dispatchEvent(new Event("input"));
  record("after_up", state.m + " " + mBox.value);
  mBox.stepDown(); mBox.dispatchEvent(new Event("input"));
  mBox.stepDown(); mBox.dispatchEvent(new Event("input"));
  record("after_down2", state.m);
  mBox.value = "2.083"; mBox.dispatchEvent(new InputEvent("input", { inputType: "insertText" }));
  record("typing_not_sent", state.m);
  mBox.dispatchEvent(new Event("change"));
  mBox.stepUp(); mBox.dispatchEvent(new Event("input"));
  record("offset_kept", state.m);
  mBox.blur();
  const button = el.querySelector("button");
  record("reset_text", button.textContent);
  record("reset_is_last", button.parentElement === el.firstElementChild.lastElementChild);
  button.click();
  record("after_reset", state.m + "," + state.b);
  record("saves", saves);
  cleanup();
  record("listeners_left", (listeners["change:m"] || []).length + (listeners["change:b"] || []).length);
} catch (e) {
  log.push("EXCEPTION=" + e.name + ": " + e.message);
}
document.getElementById("out").textContent = log.join(String.fromCharCode(10));
</script></body></html>"""


@pytest.fixture(scope="module")
def browser_log():
    m_range, b_range = p._manual_fit_slider_ranges(DxVec, FVec, dFVec,
                                                   None, None, None, None)
    state = {
        "name": "manual_fit1", "start": [0.0, 0.63],
        "x": DxVec.tolist(), "y": FVec.tolist(), "dy": dFVec.tolist(),
        "m": 0.0, "b": 0.63, "m_range": m_range, "b_range": b_range,
        "dof": len(DxVec) - 2,
        "labels": {"x_label": "x", "y_label": "y", "data_label": "Data",
                   "model_label": "model", "residual_y_label": "residual\n= data - model",
                   "chi2": True},
    }
    page = (_HARNESS.replace("__STATE__", json.dumps(state))
            .replace("__ESM__", json.dumps(p._MANUAL_FIT_ESM)))
    # One browser profile kept between runs: a brand-new profile makes
    # current Edge spend most of a minute setting itself up first.
    profile = Path(tempfile.gettempdir()) / "phys119-test-browser-profile"
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "harness.html"
        path.write_text(page, encoding="utf-8")
        result = subprocess.run(
            [BROWSER, "--headless=new", "--disable-gpu", "--no-first-run",
             "--no-default-browser-check", "--disable-extensions",
             "--disable-component-update", "--disable-background-networking",
             "--disable-sync", f"--user-data-dir={profile}",
             "--virtual-time-budget=10000", "--dump-dom", path.as_uri()],
            capture_output=True, text=True, encoding="utf-8", timeout=120,
        )
    found = re.search(r'<pre id="out">(.*?)</pre>', result.stdout, re.S)
    text = html.unescape(found.group(1)) if found else "no output"
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


@needs_browser
def test_js_runs_without_exception(browser_log):
    assert "EXCEPTION" not in browser_log, browser_log.get("EXCEPTION")
    assert browser_log["canvases"] == "2"


@needs_browser
def test_js_starting_values_shown(browser_log):
    assert browser_log["start_m_box"] == "0.00"
    assert float(browser_log["start_b_box"]) == 0.63


@needs_browser
def test_js_slider_moves_the_value(browser_log):
    assert browser_log["slider_m"] == "2.05"
    assert browser_log["slider_m_box"] == "2.05"


@needs_browser
def test_js_typed_values_are_kept_exactly(browser_log):
    assert browser_log["typed_m"] == "1.234"
    assert browser_log["typed_b"] == "0.0047"


@needs_browser
def test_js_value_beyond_range_widens_the_slider(browser_log):
    assert browser_log["wide_max"] == "50"
    assert browser_log["wide_box"] == "50.00"


@needs_browser
def test_js_blank_box_changes_nothing(browser_log):
    assert browser_log["after_blank_m"] == "50"


@needs_browser
def test_js_keys_do_not_reach_jupyterlab(browser_log):
    assert browser_log["key_leaked"] == "false"


@needs_browser
def test_js_reset_returns_to_the_start(browser_log):
    assert browser_log["after_reset"] == "0,0.63"


@needs_browser
def test_js_reset_sits_under_the_residuals_and_says_what_it_resets(browser_log):
    assert browser_log["reset_text"] == "Reset slope and intercept"
    assert browser_log["reset_is_last"] == "true"


@needs_browser
def test_js_cleanup_removes_listeners(browser_log):
    assert browser_log["listeners_left"] == "0"


@needs_anywidget
def test_ranges_can_come_from_an_unpacked_dictionary(notebook):
    fitting_boundaries = dict(m_min=1.5, m_max=2.5, b_min=-0.2, b_max=0.2)
    p.manual_fit(DxVec, FVec, dFVec, m_min=1.5, m_max=2.5, b_min=-0.2, b_max=0.2)
    p.manual_fit(DxVec, FVec, dFVec, **fitting_boundaries)
    typed, unpacked = globals()["manual_fit1"]._widget, globals()["manual_fit2"]._widget
    assert (typed.m_range, typed.b_range) == (unpacked.m_range, unpacked.b_range)
    assert (unpacked.m, unpacked.b) == (2.0, 0.0)


@needs_anywidget
def test_changing_the_dictionary_keeps_the_fit_and_its_line(notebook):
    fitting_boundaries = dict(m_min=1.5, m_max=2.5)
    for run in range(2):
        p.manual_fit(DxVec, FVec, dFVec, **fitting_boundaries)
        if run == 0:
            globals()["manual_fit1"]._widget.m = 2.2
            fitting_boundaries = dict(m_min=1.0, m_max=3.0)
    widget = globals()["manual_fit1"]._widget
    assert _fit_names() == ["manual_fit1"]
    assert widget.m == 2.2                     # saved line restored
    assert widget.m_range[:2] == [1.0, 3.0]    # new ranges used
    assert widget.start[0] == 2.0              # Reset goes to their middle


# --- Help text (step 9) ---------------------------------------------------------

def test_help_lists_every_argument():
    doc = inspect.getdoc(p.manual_fit)
    for name in ("x", "y", "dy"):
        assert f"\n  {name:<2} -- " in doc
    for name in ("m_min, m_max --", "b_min, b_max --"):
        assert name in doc


def test_help_says_nothing_about_internals():
    import pydoc
    for obj in (p.manual_fit, p.ManualFit):
        text = pydoc.render_doc(obj, renderer=pydoc.plaintext)
        for word in ("anywidget", "traitlets", "json", "MANUAL-FIT-PLAN",
                     "__dict__", "__weakref__"):
            assert word not in text, word


@needs_anywidget
def test_help_examples_run(notebook):
    doc = inspect.getdoc(p.manual_fit)
    examples = doc[doc.index("Examples:"):doc.index("The axes are labelled")]
    code = "\n".join(line[2:] for line in examples.splitlines()[1:])
    exec(code, {"manual_fit": p.manual_fit,
                "DxVec": DxVec, "FVec": FVec, "dFVec": dFVec})


# --- What is saved in the notebook, for exports ---------------------------------

@pytest.fixture
def displayed(monkeypatch):
    """Capture what manual_fit displays, as a notebook would get it."""
    shown = []

    def display(obj, raw=False):
        shown.append(obj)

    from IPython.display import HTML
    monkeypatch.setattr(p, "_notebook_display", lambda: (display, HTML))
    return shown


@needs_anywidget
def test_widget_output_carries_a_picture_and_a_sentence(notebook, displayed):
    import matplotlib.pyplot as plt
    p.manual_fit(DxVec, FVec, dFVec, m_min=1.5, m_max=2.5)
    bundle = displayed[1]                 # displayed[0] is the name line
    assert set(bundle) == {"application/vnd.jupyter.widget-view+json",
                           "image/png", "text/plain"}
    assert bundle["image/png"][:8] == b"\x89PNG\r\n\x1a\n"
    assert bundle["text/plain"].startswith("manual_fit1: a line fitted by hand")
    assert plt.get_fignums() == []        # drawn without pyplot


@needs_anywidget
def test_picture_shows_the_starting_line(notebook, displayed, monkeypatch):
    drawn = []
    original = p._manual_fit_picture
    monkeypatch.setattr(p, "_manual_fit_picture",
                        lambda x, y, dy, m, b, labels: drawn.append((m, b)) or original(x, y, dy, m, b, labels))
    p.manual_fit(DxVec, FVec, dFVec, m_min=1.5, m_max=2.5, b_min=-0.2, b_max=0.2)
    globals()["manual_fit1"]._widget.m = 2.08
    p.manual_fit(DxVec, FVec, dFVec, m_min=1.5, m_max=2.5, b_min=-0.2, b_max=0.2)
    assert drawn == [(2.0, 0.0), (2.08, 0.0)]     # after a re-run: where it was left


@needs_anywidget
def test_a_failed_picture_does_not_stop_the_widget(notebook, displayed, monkeypatch):
    def broken(*args):
        raise RuntimeError("no picture today")
    monkeypatch.setattr(p, "_manual_fit_picture", broken)
    p.manual_fit(DxVec, FVec, dFVec)
    assert "image/png" not in displayed[1]
    assert "application/vnd.jupyter.widget-view+json" in displayed[1]


# --- Fits identified by their notebook cell -------------------------------------

@needs_anywidget
def test_identical_calls_in_different_cells_are_separate_fits(notebook, monkeypatch):
    for cell in ("cell-best", "cell-lowest", "cell-highest"):
        monkeypatch.setattr(p, "_current_cell_id", lambda cell=cell: cell)
        p.manual_fit(IVec, Vr1Vec, dVr1Vec, m_min=5, m_max=15)
    assert _fit_names() == ["manual_fit1", "manual_fit2", "manual_fit3"]
    assert saved(notebook)["cell cell-lowest | call: p.manual_fit("
                           "IVec, Vr1Vec, dVr1Vec, m_min=5, m_max=15)"]["name"] == "manual_fit2"


@needs_anywidget
def test_rerunning_a_cell_finds_its_fit_even_after_a_restart(notebook, monkeypatch):
    for cell in ("cell-a", "cell-b"):
        monkeypatch.setattr(p, "_current_cell_id", lambda cell=cell: cell)
        p.manual_fit(IVec, Vr1Vec, dVr1Vec)
    globals()["manual_fit2"]._widget.m = 10.2
    monkeypatch.setattr(p, "_manual_fits", {})                # a kernel restart
    for name in _fit_names():
        del globals()[name]
    monkeypatch.setattr(p, "_current_cell_id", lambda: "cell-b")
    p.manual_fit(IVec, Vr1Vec, dVr1Vec)
    assert _fit_names() == ["manual_fit2"]
    assert globals()["manual_fit2"].m == 10.2


def test_no_kernel_means_no_cell_id():
    assert p._current_cell_id() is None



@needs_browser
def test_js_arrows_move_by_one_slider_step(browser_log):
    # the harness uses the automatic spring ranges: slope step 0.05
    assert browser_log["box_step"] == "0.05"
    assert browser_log["after_up"] == "2.13 2.13"
    assert browser_log["after_down2"] == "2.03"


@needs_browser
def test_js_arrows_keep_a_typed_offset(browser_log):
    assert browser_log["typing_not_sent"] == "2.03"
    assert browser_log["offset_kept"] == "2.133"


@needs_anywidget
def test_residual_label_is_two_lines(notebook):
    p.manual_fit(DxVec, FVec, dFVec)
    assert globals()["manual_fit1"]._widget.labels["residual_y_label"] == "residual\n= data - model"



@needs_browser
def test_js_legend_shows_live_chi2(browser_log):
    expected = np.sum(((FVec - (0 * DxVec + 0.63)) / dFVec) ** 2) / (len(DxVec) - 2)
    mantissa, exponent = p._three_sig_figs(expected)
    assert exponent is None
    assert browser_log["chi2_start"] == f"\u03c7\u00b2 = {mantissa}"


@needs_anywidget
def test_manual_fit_chi2(notebook):
    p.manual_fit(DxVec, FVec, dFVec, chi2=True)
    fit = globals()["manual_fit1"]
    assert fit._widget.labels["chi2"] is True and fit._widget.dof == len(DxVec) - 2
    fit._widget.m, fit._widget.b = 2.08, 0.0047
    expected = np.sum(((FVec - (2.08 * DxVec + 0.0047)) / dFVec) ** 2) / (len(DxVec) - 2)
    assert fit.chi2 == pytest.approx(expected, rel=1e-12)


@needs_anywidget
def test_manual_fit_chi2_needs_three_points(notebook, capsys):
    with pytest.raises(p.Phys119Error):
        p.manual_fit([1.0, 2.0], [2.0, 4.1], [0.1, 0.1], chi2=True)
    assert "needs at least 3 data points to work out chi-squared" in capsys.readouterr().err


@needs_anywidget
def test_manual_fit_picture_carries_chi2(notebook, monkeypatch):
    seen = []
    original = p._draw_data_and_model
    monkeypatch.setattr(p, "_draw_data_and_model",
                        lambda *a: seen.append(a[-1]) or original(*a))
    picture = p._manual_fit_picture(DxVec, FVec, dFVec, 2.0, 0.0,
                                    {"data_label": "Data", "model_label": "model",
                                     "x_label": "x", "y_label": "y", "chi2": True})
    assert picture[:8] == b"\x89PNG\r\n\x1a\n"
    assert seen and seen[0].startswith("$\\chi^2 = ")



@needs_browser
def test_js_slider_counts_positions_and_reaches_both_ends(browser_log):
    m_range, _ = p._manual_fit_slider_ranges(DxVec, FVec, dFVec, None, None, None, None)
    positions = round((m_range[1] - m_range[0]) / m_range[2])
    assert browser_log["slider_positions"] == f"0 {positions} 1"
    assert browser_log["at_last_position"] == "true"
    assert browser_log["at_first_position"] == "true"
