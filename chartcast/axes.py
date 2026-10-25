"""Find the value axis and the time axis from the tick labels, confirm them with an aligned
straight line when the chart draws one, and derive the plot box.

Text first: every readable chart has a column of numbers increasing upwards and a row of dates
increasing to the right, wherever they are and whether or not axis lines are drawn.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

from . import errors
from .config import Settings
from .errors import ChartError
from .ocr import Box, TextBox
from .parse import parse_date, parse_number, unwrap
from .preprocess import Prepared


@dataclass
class Line:
    pos: float    # row (horizontal) or column (vertical)
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class Axis:
    labels: list[TextBox]
    values: list          # floats (value axis) or datetimes (time axis), same order as labels
    side: str             # left | right (value axis), bottom | top (time axis)
    line: Line | None = None
    units: list[str] = field(default_factory=list)      # time axis: finest field shown per label
    corrected: list[str] = field(default_factory=list)  # doubtful labels corrected by interpolation


@dataclass
class PlotArea:
    box: Box
    value_axis: Axis | None = None
    time_axis: Axis | None = None
    h_grid: list[int] = field(default_factory=list)   # gridline rows inside the box
    v_grid: list[int] = field(default_factory=list)   # gridline columns inside the box
    y_marks: list[int] = field(default_factory=list)  # tick-mark rows on the value-axis line
    x_marks: list[int] = field(default_factory=list)  # tick-mark columns on the time-axis line
    text_height: float = 10.0


# --------------------------------------------------------------------------- straight lines

def _merge(segments: list[tuple[float, int, int]], tol: float = 3) -> list[Line]:
    """Group collinear segments (same row/column within tol) into one line each."""
    groups: list[list[tuple[float, int, int]]] = []
    for seg in sorted(segments):
        if groups and seg[0] - groups[-1][-1][0] <= tol:
            groups[-1].append(seg)
        else:
            groups.append([seg])
    return [Line(float(np.median([s[0] for s in g])), min(s[1] for s in g), max(s[2] for s in g)) for g in groups]


def find_lines(mask: np.ndarray, min_frac: float) -> tuple[list[Line], list[Line]]:
    """Long straight horizontal and vertical lines (axes, gridlines, borders) via HoughLinesP."""
    h, w = mask.shape
    min_len = int(min_frac * min(h, w))
    segs = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=max(30, min_len // 2),
                           minLineLength=min_len, maxLineGap=10)
    horiz, vert = [], []
    for x1, y1, x2, y2 in ([] if segs is None else segs.reshape(-1, 4)):  # (N,1,4) in OpenCV 4, (N,4) in 5
        if abs(y2 - y1) <= 2:
            horiz.append(((y1 + y2) / 2, min(x1, x2), max(x1, x2)))
        elif abs(x2 - x1) <= 2:
            vert.append(((x1 + x2) / 2, min(y1, y2), max(y1, y2)))
    return _merge(horiz), _merge(vert)


# --------------------------------------------------------------------------- label groups

def _clusters(items: list, key, tol: float) -> list[list]:
    """Split items sorted by key wherever consecutive keys differ by more than tol."""
    out: list[list] = []
    for it in sorted(items, key=key):
        if out and key(it) - key(out[-1][-1]) <= tol:
            out[-1].append(it)
        else:
            out.append([it])
    return out


def _longest_increasing(values: list) -> list[int]:
    """Indices of the longest strictly increasing subsequence (drops stray labels in a group)."""
    n = len(values)
    best, prev = [1] * n, [-1] * n
    for i in range(n):
        for j in range(i):
            if values[j] < values[i] and best[j] + 1 > best[i]:
                best[i], prev[i] = best[j] + 1, j
    i = int(np.argmax(best)) if n else -1
    out = []
    while i >= 0:
        out.append(i)
        i = prev[i]
    return out[::-1]


def _overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _value_line(labels: list[TextBox], V: list[Line], h: float, s: Settings) -> Line | None:
    """Vertical line beside the label column, running along it."""
    gx0, gx1 = min(t.box[0] for t in labels), max(t.box[2] for t in labels)
    gy0, gy1 = min(t.center[1] for t in labels), max(t.center[1] for t in labels)
    best, best_d = None, s.line_tol * h
    for l in V:
        d = min(abs(l.pos - gx1), abs(l.pos - gx0))
        beside = l.pos >= gx1 - 0.5 * h or l.pos <= gx0 + 0.5 * h
        # An axis line runs along all its labels; a shorter one (e.g. a tooltip crosshair) is not the axis.
        spans = l.start <= gy0 + h and l.end >= gy1 - h
        if beside and spans and d <= best_d:
            best, best_d = l, d
    return best


def _time_line(labels: list[TextBox], H: list[Line], h: float, s: Settings) -> Line | None:
    """Horizontal line just above (or below) the label row, running along it."""
    gy0, gy1 = min(t.box[1] for t in labels), max(t.box[3] for t in labels)
    gx0, gx1 = min(t.center[0] for t in labels), max(t.center[0] for t in labels)
    best, best_d = None, s.line_tol * h
    for l in H:
        d = min(abs(gy0 - l.pos), abs(l.pos - gy1))
        beside = l.pos <= gy0 + 0.5 * h or l.pos >= gy1 - 0.5 * h
        spans = l.start <= gx0 + h and l.end >= gx1 - h
        if beside and spans and d <= best_d:
            best, best_d = l, d
    return best


def _pick(groups: list[tuple[list, list]], match, s: Settings):
    """Best group: enough labels first, then one with an aligned axis line, then more labels."""
    scored = [(len(lab) >= s.min_labels, (line := match(lab)) is not None, len(lab), lab, vals, line)
              for lab, vals in groups]
    if not scored:
        return None
    return max(scored, key=lambda g: g[:3])[3:]


def _on_one_scale(pos: list[float], vals: list[float], conf: list[float],
                  sure: float = 0.9) -> list[tuple[int, float, bool]]:
    """The labels that lie on one common scale (value linear in position, or log10 of it), as
    (index, value, corrected) tuples.

    Tick labels of a real axis sit on a single, evenly stepped scale. Confidently read labels
    (OCR confidence >= sure) are the anchors: with 2+ of them the scale is known, and every less
    certain label is cross-checked against the value interpolated at its position:
      * it matches -> kept;
      * it does not, but it lies *between* anchors and exactly on the tick grid -> it is one of the
        chart's tick labels that OCR misread, so it is corrected to the grid value;
      * otherwise (off the grid, or outside the anchors, e.g. a tab button above the chart) -> it is
        not an axis label and is dropped, so it can never stretch the axis.
    With fewer than 2 anchors, a robust fit over all labels decides (needs 4+ labels; no corrections).
    """
    n = len(vals)
    p = np.asarray(pos, dtype=float)
    v = np.asarray(vals, dtype=float)
    anchors = np.asarray(conf) >= sure
    if anchors.sum() >= 2:
        mask = anchors
    elif n >= 4:
        mask = np.ones(n, bool)
    else:
        return [(i, v[i], False) for i in range(n)]

    # Label positions are reliable even when OCR misreads digits: the tick step in value units is
    # the slope times the typical pixel spacing between labels (anchors may skip a tick).
    px_step = np.median(np.diff(np.sort(p)))

    def evaluate(y):
        if mask.sum() >= 4:  # enough anchors to also catch a confident misread
            rough = np.median(np.abs(np.diff(np.sort(y[mask]))))
            fit = RANSACRegressor(LinearRegression(), min_samples=2, residual_threshold=0.25 * rough,
                                  random_state=0).fit(p[mask, None], y[mask]).estimator_
        else:
            fit = LinearRegression().fit(p[mask, None], y[mask])
        step = abs(fit.coef_[0]) * px_step
        pred = fit.predict(p[:, None])
        return np.abs(pred - y) <= 0.25 * step, pred, step

    scales = [(v, False)] + ([(np.log10(v), True)] if (v > 0).all() else [])
    (ok, pred, step), y, log = max(((evaluate(y), y, log) for y, log in scales), key=lambda r: r[0][0].sum())

    good = mask & ok
    out = [(i, v[i], False) for i in range(n) if ok[i]]
    if good.sum() >= 2 and step > 0:
        # Exact tick step from the trusted values: their gaps are whole multiples of the step,
        # the slope-based estimate only decides which multiple.
        gaps = np.diff(np.sort(y[good]))
        mult = np.round(gaps / step)
        if (mult > 0).any():
            step = float(np.median(gaps[mult > 0] / mult[mult > 0]))
        lo, hi = p[good].min(), p[good].max()
        ref = y[good][0]
        taken = {round(float(y[i]), 9) for i in range(n) if ok[i]}
        for i in range(n):
            if ok[i] or anchors[i] or not lo < p[i] < hi:
                continue
            grid = ref + round((pred[i] - ref) / step) * step
            if abs(pred[i] - grid) <= 0.25 * step and round(float(grid), 9) not in taken:
                taken.add(round(float(grid), 9))
                if log:
                    value = float(f"{10 ** grid:.3g}")                      # 3 significant digits
                else:
                    value = round(float(grid), max(0, 1 - int(np.floor(np.log10(step)))))  # step precision
                out.append((i, value, True))
    return sorted(out)


def find_value_axis(texts: list[TextBox], V: list[Line], h: float, s: Settings) -> Axis | None:
    """Numbers on (approximately) one vertical line, increasing from bottom to top."""
    cands = [(t, v) for t in texts if (v := parse_number(t.text)) is not None]
    groups, seen = [], set()
    # Axis labels share an edge: right-aligned (left axis), left-aligned (right axis) or centred.
    for key in (lambda c: c[0].box[2], lambda c: c[0].box[0], lambda c: c[0].center[0]):
        for g in _clusters(cands, key, s.align_tol * h):
            g = sorted(g, key=lambda c: -c[0].center[1])                     # bottom -> top
            g = [c for i, c in enumerate(g) if i == 0 or g[i - 1][0].center[1] - c[0].center[1] > 0.5 * h]
            scale = _on_one_scale([c[0].center[1] for c in g], [c[1] for c in g], [c[0].conf for c in g])
            g = [(g[i][0], val, fixed) for i, val, fixed in scale]
            g = [g[i] for i in _longest_increasing([c[1] for c in g])]      # values increase upwards
            ids = frozenset(id(c[0]) for c in g)
            if len(g) >= 2 and ids not in seen:
                seen.add(ids)
                groups.append(([c[0] for c in g], [(c[1], c[2]) for c in g]))
    best = _pick(groups, lambda lab: _value_line(lab, V, h, s), s)
    if not best:
        return None
    labels, vals, line = best
    corrected = [f"{t.text} → {v:g}" for t, (v, fixed) in zip(labels, vals) if fixed]
    return Axis(labels, [v for v, _ in vals], side="", line=line, corrected=corrected)


def find_time_axis(texts: list[TextBox], H: list[Line], h: float, s: Settings) -> Axis | None:
    """Dates/times on (approximately) one horizontal line, increasing from left to right."""
    cands = [(t, p) for t in texts if (p := parse_date(t.text)) is not None]
    groups = []
    for g in _clusters(cands, lambda c: c[0].center[1], s.align_tol * h):
        g = sorted(g, key=lambda c: c[0].center[0])                          # left -> right
        times = unwrap([p for _, p in g])
        scale = _on_one_scale([c[0].center[0] for c in g], [t.timestamp() for t in times],
                              [c[0].conf for c in g])
        g = [(g[i][0], g[i][1].unit, datetime.fromtimestamp(sec, timezone.utc) if fixed else times[i], fixed)
             for i, sec, fixed in scale]
        g = [g[i] for i in _longest_increasing([c[2] for c in g])]           # times increase rightwards
        if len(g) >= 2:
            groups.append(([c[0] for c in g], g))
    best = _pick(groups, lambda lab: _time_line(lab, H, h, s), s)
    if not best:
        return None
    labels, info, line = best
    corrected = [f"{t.text} → {d:%Y-%m-%d %H:%M}" for t, (_, _, d, fixed) in zip(labels, info) if fixed]
    return Axis(labels, [d for _, _, d, _ in info], side="", line=line,
                units=[u for _, u, _, _ in info], corrected=corrected)


# --------------------------------------------------------------------------- plot box

def _tick_marks(mask: np.ndarray, line: Line, vertical: bool, depth: int, lo: int, hi: int) -> list[int]:
    """Short perpendicular strokes on either side of an axis line: exact tick positions."""
    p = int(round(line.pos))
    if vertical:  # strokes are horizontal: look at columns just left/right of the line
        strips = [mask[lo:hi, max(0, p - 1 - depth):max(0, p - 1)], mask[lo:hi, p + 2:p + 2 + depth]]
        counts = [(s > 0).sum(1) for s in strips]
    else:
        strips = [mask[p + 2:p + 2 + depth, lo:hi], mask[max(0, p - 1 - depth):max(0, p - 1), lo:hi]]
        counts = [(s > 0).sum(0) for s in strips]
    marks: list[int] = []
    for c in counts:
        on = np.flatnonzero(c >= max(2, depth // 2))
        runs = np.split(on, np.flatnonzero(np.diff(on) > 1) + 1) if len(on) else []
        marks += [int(round(r.mean())) + lo for r in runs if len(r) <= 4]  # thin strokes only
    return sorted(set(marks))


def _ink_extent(mask: np.ndarray, y0: int, y1: int, x_from: int, direction: int) -> int:
    """Farthest column (left or right of x_from) that still has ink between rows y0..y1."""
    cols = np.flatnonzero(mask[max(0, y0):y1].any(0))
    cols = cols[cols > x_from] if direction > 0 else cols[cols < x_from]
    if len(cols) == 0:
        return mask.shape[1] - 1 if direction > 0 else 0
    return int(cols.max() if direction > 0 else cols.min())


def detect_axes(prep: Prepared, texts: list[TextBox], s: Settings) -> PlotArea:
    """Value axis + time axis from the labels (lines only confirm), then the plot box."""
    if not texts:
        raise ChartError(errors.NO_TEXT)
    h = float(np.median([t.height for t in texts]))
    # Letters in a row can line up into a fake "straight line" (e.g. along a label's baseline):
    # blank text-sized boxes before searching for axis lines.
    ink = prep.neutral.copy()
    for t in texts:
        if t.height <= 2 * h:
            ink[max(0, t.box[1] - 1):t.box[3] + 2, max(0, t.box[0] - 1):t.box[2] + 2] = 0
    H, V = find_lines(ink, s.min_line_frac)

    va = find_value_axis(texts, V, h, s)
    if va is None:
        raise ChartError(errors.NO_VALUES)
    ta = find_time_axis([t for t in texts if t not in va.labels], H, h, s)
    if ta is None:
        raise ChartError(errors.NO_TIMES)
    for t in va.labels:
        t.role = "value_label"
    for t in ta.labels:
        t.role = "time_label"

    # Which side is the value axis on? Relative to its line if any, else to the time labels.
    va_cx = np.mean([t.center[0] for t in va.labels])
    ref = va.line.pos if va.line else np.mean([t.center[0] for t in ta.labels])
    va.side = "left" if va_cx < ref else "right"
    ta.side = "bottom" if (ta.line is None or ta.line.pos < ta.labels[0].center[1]) else "top"

    # Vertical extent: axis lines when present, otherwise from the labels.
    ys = sorted(t.center[1] for t in va.labels)
    step = float(np.median(np.diff(ys))) if len(ys) > 1 else 2 * h
    top = va.line.start if va.line else ys[0] - 0.6 * step
    if ta.line:
        bottom = ta.line.pos
    elif va.line:
        bottom = va.line.end
    else:
        bottom = min(t.box[1] for t in ta.labels) - 0.3 * h

    # Horizontal extent: the value-axis side, then the far side (time line end, a parallel line, or ink).
    if va.side == "left":
        # First estimate only: without an axis line the search starts right after the labels;
        # the traced line sets the final horizontal extent (pipeline.extract).
        x0 = va.line.pos if va.line else max(t.box[2] for t in va.labels) + 2
        # The time-axis line defines the extent; parallel lines (frames, panel borders) only without it.
        far = [ta.line.end] if ta.line else [
            l.pos for l in V if l.pos > x0 + 4 * h and _overlap(l.start, l.end, top, bottom) > 0.5 * (bottom - top)]
        x1 = max(far) if far else _ink_extent(prep.binary, int(top), int(bottom), int(x0), +1)
    else:
        x1 = va.line.pos if va.line else min(t.box[0] for t in va.labels) - 2
        far = [ta.line.start] if ta.line else [
            l.pos for l in V if l.pos < x1 - 4 * h and _overlap(l.start, l.end, top, bottom) > 0.5 * (bottom - top)]
        x0 = min(far) if far else _ink_extent(prep.binary, int(top), int(bottom), int(x1), -1)
    H_img, W_img = prep.gray.shape
    bx0, by0 = max(0, int(round(x0))), max(0, int(round(top)))  # label-based estimates can overshoot the image
    bx1, by1 = min(W_img - 1, int(round(x1))), min(H_img - 1, int(round(bottom)))
    box = (bx0, by0, bx1, by1)

    h_grid = [int(round(l.pos)) for l in H if by0 + 3 < l.pos < by1 - 3 and l.length >= 0.6 * (bx1 - bx0)]
    v_grid = [int(round(l.pos)) for l in V if bx0 + 3 < l.pos < bx1 - 3 and l.length >= 0.6 * (by1 - by0)]
    depth = max(3, int(round(0.5 * h)))
    y_marks = _tick_marks(prep.neutral, va.line, True, depth, by0 - 5, by1 + 6) if va.line else []
    x_marks = _tick_marks(prep.neutral, ta.line, False, depth, bx0 - 5, bx1 + 6) if ta.line else []
    return PlotArea(box, va, ta, h_grid, v_grid, y_marks, x_marks, h)
