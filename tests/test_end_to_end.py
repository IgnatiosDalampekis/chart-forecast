"""Full extraction on synthetic charts vs ground truth. Needs the EasyOCR models."""
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from chartcast import errors
from chartcast.errors import ChartError
from chartcast.pipeline import extract

pytestmark = pytest.mark.ocr


def _mae_pct(series, truth):
    """MAE as % of the true value range, allowing a ±2 step alignment offset at the edges."""
    s, t = series.to_numpy(), np.asarray(truth["values"])
    best = np.inf
    for o in range(-2, 3):
        a, b = s[max(0, o):], t[max(0, -o):]
        n = min(len(a), len(b))
        best = min(best, np.abs(a[:n] - b[:n]).mean())
    return 100 * best / (t.max() - t.min())


@pytest.fixture(scope="module")
def extractions(charts):
    return [(extract(path), truth) for path, truth in charts]


def test_axes_and_plot_box(extractions):
    for ex, truth in extractions:
        x0, y0, x1, y1 = [v / ex.prep.scale for v in ex.area.box]
        tx0, ty0, tx1, ty1 = truth["plot_box"]
        assert abs(y0 - ty0) <= 2 and abs(y1 - ty1) <= 2
        # horizontally the box follows the traced line (inside the axes area, around the line)
        assert tx0 - 2 <= x0 and x1 <= tx1 + 2
        assert x0 * ex.prep.scale <= ex.px[0] and ex.px[-1] <= x1 * ex.prep.scale
        assert ex.area.value_axis.side == "left" and ex.area.time_axis.side == "bottom"


def test_extraction_accuracy(extractions):
    freq_ok, maes = 0, []
    for ex, truth in extractions:
        if ex.freq == truth["freq"]:
            freq_ok += 1
            maes.append(_mae_pct(ex.series, truth))
    assert freq_ok >= 0.8 * len(extractions)
    assert np.median(maes) < 2.0


def _chart_bytes(dates=True, values=True) -> bytes:
    idx = pd.date_range("2023-01-01", periods=60, freq="MS")
    fig, ax = plt.subplots(figsize=(9, 4), dpi=100)
    ax.plot(idx, np.sin(np.arange(60) / 4) * 50 + 500)
    if not dates:
        ax.set_xticklabels([])
    if not values:
        ax.set_yticklabels([])
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


@pytest.mark.parametrize("kwargs, message", [
    ({"dates": False}, errors.NO_TIMES),
    ({"values": False}, errors.NO_VALUES),
])
def test_missing_axis_messages(kwargs, message):
    with pytest.raises(ChartError) as e:
        extract(_chart_bytes(**kwargs))
    assert str(e.value) == message


def test_input_messages():
    with pytest.raises(ChartError, match="couldn't be read as an image"):
        extract(b"not an image")
    with pytest.raises(ChartError, match="too small"):
        extract(np.full((50, 80, 3), 255, np.uint8))
    with pytest.raises(ChartError, match="No text was found"):
        extract(np.full((400, 600, 3), 255, np.uint8))
