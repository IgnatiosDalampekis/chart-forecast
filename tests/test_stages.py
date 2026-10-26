"""Fast unit tests per stage (no OCR model needed)."""
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pytest

from chartcast.axes import Axis, Line, PlotArea, find_time_axis, find_value_axis
from chartcast.calibrate import calibrate_time, calibrate_values, finest_frequency, is_log_scale, snap_frequency
from chartcast.config import SETTINGS as S
from chartcast.forecast import check_window, fold_origins, rolling_backtest, select_and_forecast, to_steps
from chartcast.ocr import TextBox, clean_numeric, merge_words
from chartcast.parse import grouped_thousands, parse_date, parse_number, unwrap
from chartcast.pipeline import _with_year
from chartcast.preprocess import load, prepare
from chartcast.trace import point_spacing, seasonal_period, skeleton, trace
from chartcast.uncertainty import error_report

H = 12  # text height used by the mock layouts


def tb(text, cx, cy, w=30):
    return TextBox(text, (cx - w // 2, cy - H // 2, cx + w // 2, cy + H // 2), 0.9)


# Preprocessing -------------------------------------------------------------

def test_prepare_inverts_dark_and_upscales():
    img = np.zeros((300, 500, 3), np.uint8)
    cv2.line(img, (50, 250), (450, 250), (255, 255, 255), 1)
    prep = prepare(img)
    assert prep.dark and prep.scale == 2.0
    assert prep.color.shape[:2] == (600, 1000)
    assert np.median(prep.gray) > 200  # background is light after inversion


# Parsing -------------------------------------------------------------------

@pytest.mark.parametrize("text, value", [
    ("1,200", 1200), ("1.2k", 1200), ("$50", 50), ("50 €", 50), ("20%", 20), ("−3", -3), ("(5)", -5),
    ("2.5M", 2.5e6), ("3bn", 3e9), ("1e5", 1e5), ("10³", 1000), ("2×10^3", 2000), ("10⁻²", 0.01),
    ("1,5", 1.5), ("1.234.567", 1234567), ("1 234", 1234), ("€1.234,50", 1234.5),
    ("15°C", 15), ("300 MW", 300), ("5 kWh", 5), ("Jan", None), ("2023-01", None), ("abc", None),
])
def test_parse_number(text, value):
    assert parse_number(text) == value


@pytest.mark.parametrize("labels, thousands", [
    (["0", "2.000", "4.000", "6.000"], True), (["2.000", "4.000", "6.000"], True), (["0", "2,000", "4,000"], True),
    (["1.250", "1.375", "1.500"], False), (["0", "0.5", "1.0"], False), (["10", "20", "30"], False),
    (["0", "1.5", "3.000"], False),
])
def test_grouped_thousands(labels, thousands):
    assert grouped_thousands(labels) == thousands


def test_year_from_a_full_date_on_the_image():
    s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2000-09-30", "2000-10-23"]))
    labels = [tb("30 Sep", 100, 500)]
    assert _with_year(s, labels + [tb("Oct 23, 2026", 100, 50)], labels).index[0] == pd.Timestamp("2026-09-30")
    assert _with_year(s, labels + [tb("Jan 5, 2027", 100, 50)], labels).index[0] == pd.Timestamp("2026-09-30")
    assert _with_year(s, labels + [tb("Oct 23", 100, 50)], labels).index[0].year == 2000  # no year anywhere


@pytest.mark.parametrize("text, expected, unit", [
    ("2024-01-05 14:00", datetime(2024, 1, 5, 14), "minute"), ("2023-01", datetime(2023, 1, 1), "month"),
    ("13/01/2024", datetime(2024, 1, 13), "day"), ("5.1.24", datetime(2024, 1, 5), "day"),
    ("2:30 pm", datetime(2000, 1, 1, 14, 30), "minute"), ("14h", datetime(2000, 1, 1, 14), "hour"),
    ("Q3 2024", datetime(2024, 7, 1), "quarter"), ("Week 12 2024", datetime(2024, 3, 18), "week"),
    ("2024", datetime(2024, 1, 1), "year"), ("'24", datetime(2024, 1, 1), "year"),
    ("Jan 2023", datetime(2023, 1, 1), "month"), ("Jan 05", datetime(2000, 1, 5), "day"),
    ("January 5, 2024", datetime(2024, 1, 5), "day"), ("Sept 2023", datetime(2023, 9, 1), "month"),
    ("Mon", datetime(2000, 1, 3), "day"),
])
def test_parse_date(text, expected, unit):
    p = parse_date(text)
    assert p.dt.replace(tzinfo=None) == expected and p.unit == unit


@pytest.mark.parametrize("text", ["Apple", "1D", "100", "0.5", "12.00", "Daily closing price"])
def test_parse_date_rejects(text):
    assert parse_date(text) is None


@pytest.mark.parametrize("text, hour", [("12.00 p.m:", 12), ("4.00 p.m_", 16), ("4:00 PM", 16), ("9.30 am", 9)])
def test_parse_date_tolerates_ocr_punctuation(text, hour):
    assert parse_date(text).dt.hour == hour


def test_unwrap_year_less_labels():
    times = unwrap([parse_date(t) for t in ["Nov 30", "Dec 15", "Jan 01", "Feb 10"]])
    assert [d.year for d in times] == [2000, 2000, 2001, 2001]


def test_clean_numeric_and_merge_words():
    assert clean_numeric("1O0") == "100" and clean_numeric("Jan") == "Jan"
    merged = merge_words([tb("Sep", 100, 50), tb("2023", 135, 50, w=36)])
    assert [t.text for t in merged] == ["Sep 2023"]
    rotated_title = TextBox("USD", (10, 30, 24, 70), 0.9)  # tall box beside a label: not merged
    assert len(merge_words([rotated_title, tb("100", 45, 50)])) == 2


# Axis detection -------------------------------------------------------------

def _value_labels(cx, values, y0=400, step=-60):
    return [tb(str(v), cx, y0 + i * step) for i, v in enumerate(values)]


def test_value_axis_ignores_stray_numbers():
    texts = _value_labels(40, [0, 50, 100, 150]) + [tb("4512.3", 300, 20), tb("+1.2%", 380, 20), tb("37", 500, 250)]
    ax = find_value_axis(texts, [], H, S)
    assert ax.values == [0, 50, 100, 150]


def test_value_axis_on_the_right():
    texts = _value_labels(900, [170, 180, 190, 200])
    assert find_value_axis(texts, [], H, S).values == [170, 180, 190, 200]


def test_value_axis_prefers_column_with_aligned_line():
    legend = _value_labels(600, [1, 2, 3, 4])                       # numbers, but no axis line
    axis = _value_labels(40, [10, 20, 30, 40])
    line = Line(pos=60, start=150, end=420)                         # vertical line beside the x=40 column
    assert find_value_axis(legend + axis, [line], H, S).values == [10, 20, 30, 40]


def test_value_axis_accepts_two_labels():
    assert find_value_axis(_value_labels(40, [0, 100]), [], H, S).values == [0, 100]


def test_value_axis_drops_decreasing_misread():
    texts = _value_labels(40, [0, 50, 8, 150, 200])                  # '100' misread as '8'
    assert find_value_axis(texts, [], H, S).values == [0, 50, 150, 200]


def test_value_axis_drops_labels_off_the_scale():
    texts = _value_labels(40, [190, 192, 194, 196, 198]) + [tb("1", 40, 470), tb("1 M", 40, 90)]  # tab "1 M" = 1e6
    assert find_value_axis(texts, [], H, S).values == [190, 192, 194, 196, 198]


def test_low_confidence_label_is_cross_checked():
    labels = _value_labels(40, [0, 100, 300])  # evenly placed: the third should read 200
    labels[2].conf = 0.5                       # OCR was unsure about it
    assert find_value_axis(labels, [], H, S).values == [0, 100]  # interpolation says 200 -> dropped
    good = _value_labels(40, [0, 100, 200])
    good[2].conf = 0.5
    assert find_value_axis(good, [], H, S).values == [0, 100, 200]  # matches the interpolation -> kept


def test_doubtful_tick_between_anchors_is_corrected():
    labels = _value_labels(40, [0, 100, 300, 300])  # the third should read 200
    labels[2].conf = 0.5
    ax = find_value_axis(labels, [], H, S)
    assert ax.values == [0, 100, 200, 300] and ax.corrected == ["300 → 200"]
    labels = _value_labels(40, [0, 50, 8, 150, 200])  # '100' misread as '8', and OCR was unsure
    labels[2].conf = 0.4
    assert find_value_axis(labels, [], H, S).values == [0, 50, 100, 150, 200]


def test_doubtful_text_outside_or_off_grid_is_not_corrected():
    labels = _value_labels(40, [190, 192, 194, 196, 198])
    above = tb("1 M", 40, 400 - 5 * 60)          # on the grid, but above the top label (a tab button)
    off = tb("7", 40, 400 - 85)                  # between labels, but off the tick grid
    above.conf = off.conf = 0.5
    ax = find_value_axis(labels + [above, off], [], H, S)
    assert ax.values == [190, 192, 194, 196, 198] and ax.corrected == []


def test_time_axis_drops_labels_off_the_scale():
    texts = [tb(t, 100 + 150 * i, 450, w=60) for i, t in enumerate(["10:00 am", "12:00 pm", "2:00 pm", "4:00 pm"])]
    texts += [tb("2026 Q2", 760, 455, w=60)]  # a card heading on the same row
    assert [d.hour for d in find_time_axis(texts, [], H, S).values] == [10, 12, 14, 16]


def test_time_axis_with_year_wrap():
    texts = [tb(t, 100 + 150 * i, 450, w=50) for i, t in enumerate(["Dec 01", "Dec 16", "Dec 31", "Jan 15"])]
    texts += [tb("Daily load", 400, 20, w=90)]
    ax = find_time_axis(texts, [], H, S)  # evenly spaced (15 days) across the new year
    assert [d.month for d in ax.values] == [12, 12, 12, 1] and ax.values[3].year == 2001


# Calibration ----------------------------------------------------------------

def test_values_between_labels_linear():
    labels = _value_labels(40, [0, 100, 200, 300], y0=400, step=-100)
    ymap = calibrate_values(Axis(labels, [0, 100, 200, 300], "left"), [], labels)
    assert not ymap.log and ymap(150) == pytest.approx(250, abs=0.5)


def test_log_scale_detection():
    even = [400, 300, 200, 100]                                    # evenly spaced label rows
    assert is_log_scale([1, 10, 100, 1000], even, [])
    assert is_log_scale([100, 200, 400, 800], even, [])           # doubling, evenly spaced: log
    assert not is_log_scale([100, 200, 300, 400], even, [])
    assert not is_log_scale([0, 50, 100, 150], even, [])           # zero: cannot be log
    assert not is_log_scale([100, 105, 110, 115], even, [])        # narrow range: linear default
    assert is_log_scale([1, 2, 3], [300, 200, 100], [tb("Price (log scale)", 200, 20, w=120)])
    labels = _value_labels(40, [1, 10, 100, 1000], y0=400, step=-100)
    ymap = calibrate_values(Axis(labels, [1, 10, 100, 1000], "left"), [], labels)
    assert ymap.log and ymap(250) == pytest.approx(10 ** 1.5, rel=0.01)


def test_time_calibration():
    labels = [tb(t, x, 450, w=50) for t, x in [("Dec 30", 100), ("Jan 01", 200), ("Jan 03", 300)]]
    ax = find_time_axis(labels, [], H, S)
    assert calibrate_time(ax, []).slope == pytest.approx(2 * 86400 / 100)


def test_snap_frequency():
    assert snap_frequency(3600) == "h"
    assert snap_frequency(86400 * 1.2) == "D"
    assert snap_frequency(86400 * 30) == "MS"
    assert snap_frequency(290) == "5min"
    assert snap_frequency(290, finest="h") == "h"  # day-level labels: no sub-hourly steps


def test_finest_frequency_from_labels():
    assert finest_frequency(["minute", "minute"]) == "min"   # '12:00 p.m.' labels
    assert finest_frequency(["day", "day"]) == "h"           # 'Jan 05' labels


# Tracing ----------------------------------------------------------------------

def _drawn_line(n=60, spacing=15.0, thickness=3, seed=0):
    rng = np.random.default_rng(seed)
    xs = 20 + spacing * np.arange(n)
    ys = 200 + 60 * np.sin(np.arange(n) / 4) + rng.normal(0, 10, n)
    img = np.zeros((420, int(xs[-1]) + 40), np.uint8)
    pts = np.stack([xs, ys], 1).round().astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], False, 255, thickness, lineType=cv2.LINE_8)
    return img, xs, ys


def test_skeleton_trace_recovers_line():
    mask, xs, ys = _drawn_line()
    px, py = trace(mask, skeleton(mask))
    err = np.abs(np.interp(xs, px, py) - np.round(ys))
    assert np.median(err) < 1.0 and err.max() < 6


def test_point_spacing_from_kinks(tmp_path):
    import json

    from chartcast.line import isolate_lines
    from chartcast.synth import make_chart

    path = make_chart("sales", 101, tmp_path)  # 60 monthly points, ~14.8 px apart
    prep = prepare(load(path))
    box = tuple(int(round(v * prep.scale)) for v in json.loads(path.with_suffix(".json").read_text())["plot_box"])
    mask = isolate_lines(prep, PlotArea(box), [])[0]
    px, py = trace(mask, skeleton(mask))
    assert point_spacing(py) == pytest.approx(len(px) / 59, rel=0.05)


def test_point_spacing_abstains_when_unsure():
    mask, _, _ = _drawn_line(spacing=15.0)  # aliased drawing: weak kinks
    px, py = trace(mask, skeleton(mask))
    sp = point_spacing(py)
    assert sp is None or sp == pytest.approx(15.0, rel=0.05)


def test_seasonal_period():
    s = pd.Series(np.sin(2 * np.pi * np.arange(240) / 24) + 0.01 * np.arange(240))
    assert seasonal_period(s) == 24


# Forecasting and uncertainty ---------------------------------------------------

@pytest.fixture
def seasonal_series():
    rng = np.random.default_rng(1)
    t = np.arange(120)
    y = 100 + 0.2 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1, 120)
    return pd.Series(y, index=pd.date_range("2015-01-01", periods=120, freq="MS"))


def test_backtest_shapes_and_no_leakage(seasonal_series):
    bt = rolling_backtest(seasonal_series, "seasonal_naive", horizon=6, period=12, folds=10)
    assert bt.forecasts.shape == bt.actuals.shape == (10, 6)
    t = bt.origins[0]  # seasonal naive forecasts only use data before each origin
    assert np.allclose(bt.forecasts[0], seasonal_series.to_numpy()[t - 12:t - 6])


def test_to_steps():
    assert to_steps(1, "days", "h") == 24
    assert to_steps(1, "weeks", "D") == 7
    assert to_steps(2, "years", "MS") == 24
    assert to_steps(1, "hours", "D") == 0  # shorter than one sample


def test_sliding_window_uses_exactly_the_window(seasonal_series):
    y = seasonal_series.to_numpy()
    bt = rolling_backtest(seasonal_series, "seasonal_naive", horizon=6, period=12, window=24, folds=5)
    t = bt.origins[0]
    assert bt.window == 24 and t == 24
    assert np.allclose(bt.forecasts[0], y[t - 12:t - 6])
    short = rolling_backtest(seasonal_series, "seasonal_naive", horizon=6, period=12, window=12, folds=5)
    assert np.allclose(short.forecasts[0], y[short.origins[0] - 1])  # < 2 seasons in window: no seasonality


@pytest.mark.parametrize("horizon, window, msg", [(0, 10, "shorter than one sample"),
                                                  (6, 1, "shorter than two samples"),
                                                  (60, 70, "exceed")])
def test_check_window_errors(horizon, window, msg):
    with pytest.raises(ValueError, match=msg):
        check_window(120, horizon, window)


def test_step_places_folds_back_from_the_latest_data(seasonal_series):
    bt = rolling_backtest(seasonal_series, "naive", horizon=6, period=12, window=60, step=10)
    assert bt.origins[-1] == 120 - 6                       # the last fold tests the most recent data
    assert np.all(np.diff(bt.origins) == 10) and bt.origins[0] >= 60
    assert list(fold_origins(120, 6, 60, step=54)) == [60, 114]  # largest step that still leaves 2 folds


@pytest.mark.parametrize("step, msg", [(0, "shorter than one sample"), (55, "only 1 validation fold"),
                                       (1, None)])
def test_check_step(step, msg):
    if msg:
        with pytest.raises(ValueError, match=msg):
            check_window(120, 6, 60, step)
    else:
        check_window(120, 6, 60, step)  # 55 folds: fine
    with pytest.raises(ValueError, match="at most 200"):
        check_window(1000, 6, 60, 1)    # 935 folds


def test_mean_error_and_p_incorrect_can_pick_different_models():
    from chartcast.forecast import Backtest

    actual = np.full((5, 1), 100.0)
    a = Backtest("a", np.arange(5), actual + np.array([[1], [1], [1], [1], [30]]), actual, 10)  # mostly close
    b = Backtest("b", np.arange(5), actual + 9, actual, 10)                                     # always 9% off
    assert a.mean_error < b.mean_error                          # 6.8% vs 9%: A has the lower mean error
    assert a.p_incorrect(0.10) > b.p_incorrect(0.10) == 0.0     # 20% vs 0%: B is never beyond 10%


def test_seasonal_model_beats_naive(seasonal_series):
    fc, best, results = select_and_forecast(seasonal_series, 6, 12)
    assert results["seasonal_naive"].mean_error < results["naive"].mean_error
    assert best.model != "naive"
    assert fc.index[0] == seasonal_series.index[-1] + pd.offsets.MonthBegin()


def test_error_report(seasonal_series):
    bt = rolling_backtest(seasonal_series, "naive", horizon=6, period=12)
    r = error_report(bt, tolerance=0.05)
    assert r.p_incorrect.shape == (6,)
    assert np.all((r.ci_low <= r.p_incorrect) & (r.p_incorrect <= r.ci_high))
    assert np.allclose(r.p_incorrect, (bt.errors > 0.05).mean(0))
