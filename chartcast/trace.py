"""Stage 7: skeleton tracing and conversion to a regular time series."""
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from scipy.signal import find_peaks
from skimage.morphology import skeletonize

from .calibrate import FREQS, AxisMap, snap_frequency

_NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def skeleton(mask: np.ndarray, min_spur: int = 5) -> np.ndarray:
    """Close small gaps, skeletonise to 1 px and prune spurs shorter than min_spur."""
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    skel = skeletonize(closed)

    nb = convolve(skel.astype(int), _NEIGHBOURS, mode="constant") * skel
    ends = nb == 1
    branches = cv2.dilate((nb >= 3).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    # Pieces between branch points; a short piece that ends in an endpoint is a spur.
    n, labels = cv2.connectedComponents((skel & ~branches).astype(np.uint8), connectivity=8)
    sizes = np.bincount(labels.ravel(), minlength=n)
    has_end = np.zeros(n, bool)
    has_end[np.unique(labels[ends])] = True
    touches_branch = np.zeros(n, bool)
    touches_branch[np.unique(labels[cv2.dilate(branches.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0])] = True
    spur = (sizes < min_spur) & has_end & touches_branch
    spur[0] = False
    return skel & ~spur[labels]


def trace(mask: np.ndarray, skel: np.ndarray, max_gap: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Per-column y with continuity, sub-pixel refined on the original mask; short gaps interpolated."""
    cols = np.flatnonzero(skel.any(0))
    if len(cols) == 0:
        raise ValueError("no line pixels to trace")
    px = np.arange(cols[0], cols[-1] + 1)
    py = np.full(len(px), np.nan)
    tops, bots = np.full(len(px), np.nan), np.full(len(px), np.nan)
    prev = None
    for i, x in enumerate(px):
        ys = np.flatnonzero(skel[:, x])
        if len(ys) == 0:
            continue
        y = ys[np.argmin(np.abs(ys - prev))] if prev is not None else int(np.median(ys))
        # Sub-pixel: centre of the contiguous run of the original mask containing y.
        col = mask[:, x] > 0
        top, bot = y, y
        while top > 0 and col[top - 1]:
            top -= 1
        while bot < len(col) - 1 and col[bot + 1]:
            bot += 1
        if col[y]:
            py[i], tops[i], bots[i] = (top + bot) / 2, top, bot
        else:
            py[i] = float(y)
        prev = y

    # At a sharp peak/trough the column run covers both slopes, so its centre undershoots the tip:
    # use the run edge, offset by half the line thickness, instead.
    runs = bots - tops + 1
    thick = np.nanmedian(runs)
    for i in range(1, len(px) - 1):
        if not runs[i] > 1.5 * thick or np.isnan(py[i - 1]) or np.isnan(py[i + 1]):
            continue
        if py[i - 1] > py[i] and py[i + 1] > py[i]:      # local max (image y grows downwards)
            py[i] = tops[i] + (thick - 1) / 2
        elif py[i - 1] < py[i] and py[i + 1] < py[i]:    # local min
            py[i] = bots[i] - (thick - 1) / 2

    s = pd.Series(py).interpolate(limit=max(1, int(max_gap * len(px))), limit_area="inside")
    ok = s.notna().to_numpy()
    return px[ok], s.to_numpy()[ok]


def point_spacing(py: np.ndarray, min_points: int = 20, min_px: float = 3.0, min_snr: float = 5.0) -> float | None:
    """Pixels per data point, from the periodicity of the line's kinks.

    Line charts draw straight segments between samples, so the slope changes at every sample and
    the samples sit on a regular pixel grid: |second difference| has a sharp spectral peak at
    1 / spacing. Returns None when that peak is not clearly above the noise or points are too
    dense to resolve.
    """
    d2 = np.abs(np.diff(py, 2))
    if len(d2) < 4 * min_points:
        return None
    spec = np.abs(np.fft.rfft(d2 - d2.mean()))
    f = np.fft.rfftfreq(len(d2))
    band = (f >= min_points / len(d2)) & (f <= 1 / min_px)  # excludes seasonal-envelope harmonics
    if not band.any():
        return None
    i = np.argmax(np.where(band, spec, 0))
    if spec[i] < min_snr * np.median(spec[band]):
        return None
    return float(1 / f[i])


def infer_frequency(px: np.ndarray, py: np.ndarray, xmap: AxisMap, finest: str = "min",
                    scale: float = 1.0) -> tuple[str, str]:
    """Sampling frequency (not finer than `finest`) and how it was found ('kinks' or 'resolution').

    Measured at the original image resolution: the upscaling done for OCR adds pixels (and smooths
    corners) but no information about how many samples the chart was drawn from.
    """
    grid = np.arange(px[0], px[-1] + 1e-9, scale)       # one column per original pixel
    py = np.interp(grid, px, py)
    sec_per_px = abs(xmap.slope) * scale
    spacing = point_spacing(py)
    if spacing is not None:
        return snap_frequency(sec_per_px * spacing, finest), "kinks"
    # Points too dense to see: assume the finest standard step the plot width can still show.
    span = sec_per_px * (len(grid) - 1)
    for f in FREQS:
        if FREQS[f] >= FREQS[finest] and span / FREQS[f] <= len(grid):
            return f, "resolution"
    return "YS", "resolution"


def _anchor(t: pd.Timestamp, freq: str) -> pd.Timestamp:
    """Nearest timestamp to t on the regular grid of `freq` (midnight-aligned for daily and coarser)."""
    step = pd.tseries.frequencies.to_offset(freq)
    if FREQS[freq] < FREQS["D"]:  # intraday: round to the step (e.g. 09:31:40 -> 09:30 for 5min)
        return t.round(freq)
    grid = pd.date_range(t - 3 * step, t + 3 * step, freq=freq, normalize=True)
    return grid[np.argmin(np.abs((grid - t).total_seconds()))]


def to_series(px: np.ndarray, py: np.ndarray, xmap: AxisMap, ymap: AxisMap, freq: str | None) -> pd.Series:
    """Apply calibrations and resample to one point per time step."""
    secs = xmap(px)
    values = ymap(py)
    if freq is None:
        return pd.Series(values, index=pd.to_datetime(secs, unit="s"), name="value")
    start = _anchor(pd.to_datetime(secs[0], unit="s"), freq)
    end = _anchor(pd.to_datetime(secs[-1], unit="s"), freq)
    idx = pd.date_range(start, end, freq=freq)
    grid = (idx - pd.Timestamp(0)).total_seconds().to_numpy()
    return pd.Series(np.interp(grid, secs, values), index=idx, name="value")


def seasonal_period(series: pd.Series, max_lag: int | None = None, min_acf: float = 0.3) -> int | None:
    """Dominant period from the autocorrelation function, if any."""
    from statsmodels.tsa.stattools import acf

    y = series.to_numpy(dtype=float)
    if len(y) < 8:
        return None
    t = np.arange(len(y))
    y = y - np.polyval(np.polyfit(t, y, 1), t)  # remove linear trend
    r = acf(y, nlags=max_lag or len(y) // 2, fft=True)
    peaks, props = find_peaks(r, height=min_acf)
    peaks = peaks[peaks >= 2]
    if len(peaks) == 0:
        return None
    return int(peaks[np.argmax(r[peaks])])
