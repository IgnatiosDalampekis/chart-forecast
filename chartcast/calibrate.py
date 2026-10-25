"""Map pixels to values and timestamps from the axis labels (linear by default, log on evidence)."""
import re
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

from .axes import Axis
from .ocr import TextBox

# Standard sampling frequencies (pandas alias -> approximate seconds per step).
FREQS = {"min": 60, "5min": 300, "15min": 900, "30min": 1800, "h": 3600,
         "D": 86400, "W": 604800, "MS": 2629746, "QS": 7889238, "YS": 31556952}


@dataclass
class AxisMap:
    slope: float
    intercept: float
    log: bool = False   # fit was on log10(value)
    r2: float = 0.0
    inliers: int = 0

    def __call__(self, px: np.ndarray) -> np.ndarray:
        v = self.slope * np.asarray(px, dtype=float) + self.intercept
        return 10 ** v if self.log else v


def _robust_fit(px: np.ndarray, v: np.ndarray, log: bool = False) -> AxisMap:
    """Least squares through the label anchors; RANSAC from 4 labels on, to drop misreads."""
    X = px.reshape(-1, 1)
    if len(px) >= 4:
        model = RANSACRegressor(LinearRegression(), min_samples=2, random_state=0).fit(X, v)
        mask, lin = model.inlier_mask_, model.estimator_
    else:
        mask, lin = np.ones(len(px), bool), LinearRegression().fit(X, v)
    r2 = lin.score(X[mask], v[mask]) if mask.sum() > 2 else 1.0
    return AxisMap(float(lin.coef_[0]), float(lin.intercept_), log, float(r2), int(mask.sum()))


def _snap(pos: float, marks: list[int], tol: float) -> float:
    """Move a label centre onto the nearest tick mark / gridline if one is within tol px."""
    if not marks:
        return pos
    m = min(marks, key=lambda m: abs(m - pos))
    return float(m) if abs(m - pos) <= tol else pos


_LOG_WORDS = re.compile(r"\blog(arithmic)?\b|\blog[- ]?scale\b", re.I)


def is_log_scale(values: list[float], pixels: list[float], texts: list[TextBox]) -> bool:
    """Linear unless there is evidence: the chart says 'log', or the label positions fit
    log10(value) clearly better than value (e.g. 1, 10, 100, 1000 evenly spaced)."""
    if any(_LOG_WORDS.search(t.text) for t in texts):
        return True
    v, p = np.asarray(values, dtype=float), np.asarray(pixels, dtype=float)
    if len(v) < 3 or (v <= 0).any() or v.max() / v.min() < 3:  # too little evidence to tell
        return False
    r2 = lambda y: np.corrcoef(p, y)[0, 1] ** 2
    return r2(np.log10(v)) > r2(v) + 0.01


def calibrate_values(axis: Axis, marks: list[int], texts: list[TextBox]) -> AxisMap:
    """Pixel row -> value from the value-axis labels."""
    py = np.array([_snap(t.center[1], marks, t.height / 2) for t in axis.labels])
    v = np.asarray(axis.values, dtype=float)
    if is_log_scale(axis.values, py, texts):
        return _robust_fit(py, np.log10(v), log=True)
    return _robust_fit(py, v)


def calibrate_time(axis: Axis, marks: list[int]) -> AxisMap:
    """Pixel column -> epoch seconds from the time-axis labels."""
    px = np.array([_snap(t.center[0], marks, (t.box[2] - t.box[0]) / 2) for t in axis.labels])
    secs = np.array([d.timestamp() for d in axis.values])
    return _robust_fit(px, secs)


def finest_frequency(units: list[str]) -> str:
    """Finest plausible sampling step given the time labels: sub-hourly steps only when the labels
    show clock times ('12:00 p.m.', '14:00'); otherwise hourly at the finest."""
    return "min" if any(u in ("second", "minute", "hour") for u in units) else "h"


def snap_frequency(seconds: float, finest: str = "min") -> str:
    """Nearest standard frequency (in log space, not finer than `finest`) to a step in seconds."""
    allowed = [f for f in FREQS if FREQS[f] >= FREQS[finest]]
    return min(allowed, key=lambda f: abs(np.log(FREQS[f]) - np.log(max(seconds, 1.0))))
