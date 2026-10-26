"""End-to-end orchestration: image -> series. Fully automatic: the only input is the image."""
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd

from .axes import Axis, PlotArea, detect_axes
from .calibrate import calibrate_time, calibrate_values, finest_frequency
from . import errors
from .config import SETTINGS
from .errors import ChartError
from .line import isolate_lines
from .ocr import NUMBER_CHARS, TextBox, read_text, reread
from .parse import PLACEHOLDER_YEAR, parse_date, parse_number
from .preprocess import Prepared, load, prepare
from .trace import infer_frequency, seasonal_period, skeleton, to_series, trace


@dataclass
class Extraction:
    series: pd.Series
    area: PlotArea
    texts: list[TextBox]
    freq: str | None
    period: int | None
    confidence: dict = field(default_factory=dict)  # per-stage quality checks
    prep: Prepared | None = None
    px: np.ndarray | None = None
    py: np.ndarray | None = None
    n_series: int = 1


def _reread_axis(img, texts: list[TextBox], axis: Axis, h: float, numbers: bool, s) -> list[TextBox]:
    """OCR the axis label strip again (upscaled); keep it if more labels parse."""
    xs0, ys0, xs1, ys1 = zip(*(t.box for t in axis.labels))
    H, W = img.shape[:2]
    # The strip spans the whole column (value axis) or row (time axis): labels missed so far may lie
    # beyond the ones already found.
    if numbers:
        region = (int(min(xs0) - h), 0, int(max(xs1) + 0.5 * h), H)
    else:
        region = (0, int(min(ys0) - 0.5 * h), W, int(max(ys1) + 0.5 * h))
    inside = lambda t: region[0] <= t.center[0] <= region[2] and region[1] <= t.center[1] <= region[3]
    parse = parse_number if numbers else parse_date
    second = reread(img, region, s.min_conf, NUMBER_CHARS if numbers else None)
    first = [t for t in texts if inside(t)]
    if sum(parse(t.text) is not None for t in second) > sum(parse(t.text) is not None for t in first):
        return [t for t in texts if not inside(t)] + second
    return texts


def _decode(image) -> np.ndarray:
    """Path, encoded bytes (an upload) or BGR array -> BGR array."""
    if isinstance(image, np.ndarray):
        return image
    if isinstance(image, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ChartError(errors.UNREADABLE)
        return img
    return load(image)


def _with_year(series: pd.Series, texts: list[TextBox], labels: list[TextBox]) -> pd.Series:
    """Time labels without a year ('30 Sep') get a placeholder year. If another text on the image is
    a full date ('Oct 23, 2026'), the chart is taken to end on or before the latest such date."""
    own = {id(t) for t in labels}
    dated = [p.dt for t in texts if id(t) not in own and (p := parse_date(t.text)) is not None
             and p.wrap is None and p.unit in ("day", "month") and p.dt.year != PLACEHOLDER_YEAR]
    if not dated:
        return series
    ref, last = pd.Timestamp(max(dated)).tz_convert(None), series.index[-1]
    years = ref.year - last.year
    if last + pd.DateOffset(years=years) > ref + pd.Timedelta(days=1):  # e.g. a December chart dated January
        years -= 1
    return series.set_axis(series.index + pd.DateOffset(years=years))


def extract(image) -> Extraction:
    """Stages 1–7 on a path, encoded image bytes or a BGR array.

    Raises ChartError, with a message for the user, when the image lacks what a forecast needs.
    """
    s = SETTINGS
    img = _decode(image)
    if min(img.shape[:2]) < 100:
        raise ChartError(errors.TOO_SMALL)
    prep = prepare(img, s.min_width)
    conf: dict = {}

    texts = read_text(prep.color, s.min_conf, s.ocr_upscale)
    area = detect_axes(prep, texts, s)
    if s.reread:
        texts = _reread_axis(prep.color, texts, area.value_axis, area.text_height, True, s)
        texts = _reread_axis(prep.color, texts, area.time_axis, area.text_height, False, s)
        area = detect_axes(prep, texts, s)
    va, ta = area.value_axis, area.time_axis
    conf["value_axis"] = {"side": va.side, "labels": len(va.labels), "line": va.line is not None,
                          "corrected": va.corrected}
    conf["time_axis"] = {"side": ta.side, "labels": len(ta.labels), "line": ta.line is not None,
                         "corrected": ta.corrected}

    ymap = calibrate_values(va, area.y_marks + area.h_grid, texts)
    xmap = calibrate_time(ta, area.x_marks + area.v_grid)
    conf["y_calibration"] = {"r2": round(ymap.r2, 4), "inliers": ymap.inliers, "log": ymap.log}
    conf["x_calibration"] = {"r2": round(xmap.r2, 4), "inliers": xmap.inliers}

    masks = isolate_lines(prep, area, texts)
    if not masks:
        raise ChartError(errors.NO_LINE)
    try:
        px, py = trace(masks[0], skeleton(masks[0]))
    except ValueError:  # mask without traceable pixels
        raise ChartError(errors.NO_LINE) from None
    x0, y0, x1, y1 = area.box
    conf["column_coverage"] = round(len(px) / max(x1 - x0, 1), 3)
    # Cross-check: labels and axis lines give a first estimate of the x-range, but a time axis
    # usually ends where the data ends. The traced line sets the final range (plus a little room).
    pad = int(round(0.5 * area.text_height))
    area.box = (max(x0, int(px[0]) - pad), y0, min(x1, int(px[-1]) + pad), y1)

    freq, how = infer_frequency(px, py, xmap, finest_frequency(ta.units), prep.scale)
    conf["frequency_source"] = how
    series = to_series(px, py, xmap, ymap, freq)
    if series.index[0].year == PLACEHOLDER_YEAR:
        series = _with_year(series, texts, ta.labels)
    if len(series) < s.min_samples:
        raise ChartError(errors.TOO_FEW.format(n=len(series)))
    return Extraction(series, area, texts, freq, seasonal_period(series), conf, prep, px, py, len(masks))


def overlay(ex: Extraction) -> np.ndarray:
    """Original image with plot box (orange), value labels (green), time labels (blue),
    detected axis lines (red) and the traced series (magenta). RGB."""
    img = ex.prep.color.copy()
    if ex.prep.dark:
        img = 255 - img
    x0, y0, x1, y1 = ex.area.box
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 140, 255), 2)
    for axis, color in ((ex.area.value_axis, (0, 180, 0)), (ex.area.time_axis, (255, 120, 0))):
        for t in axis.labels:
            cv2.rectangle(img, t.box[:2], t.box[2:], color, 2)
    va, ta = ex.area.value_axis.line, ex.area.time_axis.line
    if va:
        cv2.line(img, (int(va.pos), va.start), (int(va.pos), va.end), (0, 0, 230), 2)
    if ta:
        cv2.line(img, (ta.start, int(ta.pos)), (ta.end, int(ta.pos)), (0, 0, 230), 2)
    pts = np.stack([ex.px, ex.py], axis=1).round().astype(np.int32)
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (255, 0, 255), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


_UNIT_FORMAT = {"second": "%H:%M:%S", "minute": "%H:%M", "hour": "%H:%M", "day": "%d %b %Y", "week": "%d %b %Y",
                "month": "%b %Y", "quarter": "%b %Y", "year": "%Y"}


def _time_text(label: str, dt) -> str:
    """The parsed meaning of a time label (fields the label did not show are left out)."""
    p = parse_date(label)
    fmt = _UNIT_FORMAT.get(p.unit, "%Y-%m-%d") if p else "%Y-%m-%d"
    if p and p.wrap in ("year", "week") and fmt.endswith(" %Y"):
        fmt = fmt[:-3]
    return dt.strftime(fmt)


def reconstruct(ex: Extraction, size: tuple[int, int] | None = None):
    """Clean redraw of what the algorithm understood: the two axes, the tick labels it read (with the
    values it parsed) and the traced line, cropped to the chart area. Returns a Figure.
    size=(width, height) in pixels stretches the chart to fill exactly that size (e.g. the input image's)."""
    from matplotlib.figure import Figure

    x0, y0, x1, y1 = ex.area.box
    va, ta = ex.area.value_axis, ex.area.time_axis
    h = ex.area.text_height
    # The time axis is drawn only over the traced line's range: labels beyond it are left out.
    times = [(t, d) for t, d in zip(ta.labels, ta.values) if x0 - h <= t.center[0] <= x1 + h]
    m = 1.5 * h  # the chart itself: the box, its labels and a small margin
    labels = va.labels + [t for t, _ in times]
    cx0 = min([x0] + [t.box[0] for t in labels]) - m
    cx1 = max([x1] + [t.box[2] for t in labels]) + m
    cy0 = min([y0] + [t.box[1] for t in labels]) - m
    cy1 = max([y1] + [t.box[3] for t in labels]) + m
    w, hgt = size if size else (cx1 - cx0, cy1 - cy0)
    fig = Figure(figsize=(w / 100, hgt / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(cx0, cx1)
    ax.set_ylim(cy1, cy0)
    ax.axis("off")
    font = max(6.0, ex.area.text_height * 0.72 * 0.8)  # px -> pt at 100 dpi, slightly smaller
    ink = "#444441"

    vx = va.line.pos if va.line else (x0 if va.side == "left" else x1)
    ty = ta.line.pos if ta.line else y1
    # The time axis runs from the value axis (they meet at the corner) to the end of the line.
    tx0, tx1 = (min(vx, x0), x1) if va.side == "left" else (x0, max(vx, x1))
    ax.plot([vx, vx], [y0, ty], color=ink, lw=1.2)
    ax.plot([tx0, tx1], [ty, ty], color=ink, lw=1.2)
    tick = 0.4 * ex.area.text_height
    for t, v in zip(va.labels, va.values):
        cy = t.center[1]
        ax.plot([tx0, tx1], [cy, cy], color="#D3D1C7", lw=0.6, zorder=0)
        sign = -1 if va.side == "left" else 1
        ax.plot([vx, vx + sign * tick], [cy, cy], color=ink, lw=1)
        ax.text(vx + sign * 1.5 * tick, cy, f"{v:g}", ha="right" if va.side == "left" else "left",
                va="center", fontsize=font, color=ink)
    for t, d in times:
        cx = t.center[0]
        ax.plot([cx, cx], [ty, ty + tick], color=ink, lw=1)
        ax.text(cx, ty + 1.5 * tick, _time_text(t.text, d), ha="center", va="top", fontsize=font, color=ink)
    ax.plot(ex.px, ex.py, color="#378ADD", lw=2)
    return fig
