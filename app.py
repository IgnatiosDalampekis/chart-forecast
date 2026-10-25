"""Streamlit UI: pick a test sample or upload a chart -> see what was read -> configure and run the forecast.

    streamlit run app.py
"""
import base64
import threading
import html
import json
from pathlib import Path
from urllib.parse import quote

import cv2

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.dates as mdates
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from chartcast.calibrate import FREQS
from chartcast.errors import ChartError
from chartcast.ocr import ocr_ready, warm_up
from chartcast.forecast import MAX_FOLDS, MODELS, UNITS, check_window, fold_origins, select_and_forecast, to_steps
from chartcast.pipeline import extract, overlay, reconstruct
from chartcast.uncertainty import error_report

SAMPLES = Path(__file__).parent / "samples"
FREQ_NAMES = {"min": "every minute", "5min": "every 5 minutes", "15min": "every 15 minutes",
              "30min": "every 30 minutes", "h": "hourly", "D": "daily", "W": "weekly",
              "MS": "monthly", "QS": "quarterly", "YS": "yearly"}
MODEL_NAMES = {"auto_error": "Auto (lowest mean error in validation)",
               "best_pinc": "Lowest P(incorrect) in validation", "naive": "Naive (last value)",
               "seasonal_naive": "Seasonal naive", "holt_winters": "Holt-Winters", "arima": "ARIMA"}
DEFAULT_UNIT = {"min": "hours", "5min": "hours", "15min": "hours", "30min": "days", "h": "days",
                "D": "weeks", "W": "months", "MS": "years", "QS": "years", "YS": "years"}
PER_STEP = {"min": "one value every minute", "5min": "one value every 5 minutes", "15min": "one value every 15 minutes",
            "30min": "one value every 30 minutes", "h": "one value per hour", "D": "one value per day",
            "W": "one value per week", "MS": "one value per month", "QS": "one value per quarter",
            "YS": "one value per year"}
FREQ_SHORT = {"min": "every min", "5min": "every 5 min", "15min": "every 15 min", "30min": "every 30 min"}
INTRADAY = ("min", "5min", "15min", "30min", "h")
_STATS_CSS = """<style>
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px 20px;margin:4px 0 14px}
.stats .l{font-size:14px;opacity:.75}
.stats .v{font-size:clamp(1.25rem,2.1vw,2.1rem);line-height:1.25;white-space:normal;overflow-wrap:anywhere}
.stats .short{display:none}
@media (max-width:1100px){.stats .long{display:none}.stats .short{display:inline}}
</style>"""


def accuracy_check(series: pd.Series, truth_file: Path) -> tuple[np.ndarray, float]:
    """Plot of the true series (the data the synthetic chart was drawn from) with the extracted one on top,
    and the mean absolute error as % of the true value range."""
    t = json.loads(truth_file.read_text())
    truth = pd.Series(t["values"], index=pd.to_datetime(t["timestamps"]))
    ext = series.copy()
    if ext.index[0].year == 2000 and truth.index[0].year != 2000:  # labels had no year: use the true one
        ext.index = ext.index.map(lambda d: d.replace(year=d.year + truth.index[0].year - 2000))
    secs = lambda idx: (idx - pd.Timestamp(0)).total_seconds().to_numpy()
    expected = np.interp(secs(ext.index), secs(truth.index), truth.to_numpy())
    err = float(np.mean(np.abs(ext.to_numpy() - expected)) / (truth.max() - truth.min()))
    fig = Figure(figsize=(11, 3.2), dpi=100)
    ax = fig.add_subplot()
    ax.plot(truth.index, truth.values, color="0.6", lw=3, label="true data (used to draw the chart)")
    ax.plot(ext.index, ext.values, color="#378ADD", lw=1.2, label="extracted from the image")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2, frameon=False)  # above the plot, off the lines
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=9))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return to_rgb(fig), err


def duration(seconds: float) -> str:
    """Seconds as the largest fitting unit: 1.5 years, 3 weeks, 24 hours."""
    for unit in ("years", "months", "weeks", "days", "hours", "minutes"):
        amount = seconds / UNITS[unit]
        if amount >= 1:
            amount = round(amount, 1) if amount < 10 else round(amount)
            return f"{amount:g} {unit[:-1] if amount == 1 else unit}"
    return f"{seconds:.0f} seconds"


def stat_row(items: list[tuple[str, str, str | None]]):
    """Label/value pairs like st.metric, but values wrap instead of being cut with '…';
    an optional short value replaces the long one on narrow screens."""
    cells = "".join(f'<div><div class="l">{label}</div><div class="v"><span class="long">{long}</span>'
                    f'<span class="short">{short or long}</span></div></div>' for label, long, short in items)
    st.markdown(_STATS_CSS + f'<div class="stats">{cells}</div>', unsafe_allow_html=True)


def possible_units(freq: str) -> list[str]:
    """Units no finer than the sampling: the finest is the largest unit that fits in one sample
    (5-minute data -> minutes and up, daily -> days and up, quarterly -> months and up)."""
    finest = max((u for u in UNITS if UNITS[u] <= FREQS[freq]), key=UNITS.get, default="minutes")
    return [u for u in UNITS if UNITS[u] >= UNITS[finest]]


def time_format(index: pd.DatetimeIndex, freq: str) -> str:
    """How timestamps of this chart are shown. Fields the labels did not give are left out:
    clock-time-only labels (one day) show just the time, labels without a year show no year."""
    placeholder = index[0].year == 2000  # parse.py's placeholder for labels without a year
    if freq in INTRADAY:
        if placeholder and index[-1].normalize() == index[0].normalize():
            return "%H:%M"
        return "%d %b %H:%M" if placeholder else "%Y-%m-%d %H:%M"
    return "%d %b" if placeholder else "%Y-%m-%d"

st.set_page_config(page_title="chartcast", layout="wide")


@st.cache_resource
def start_ocr_warm_up() -> threading.Thread:
    """Once per app process: load the OCR model in the background while the user picks an image."""
    thread = threading.Thread(target=warm_up, daemon=True)
    thread.start()
    return thread


start_ocr_warm_up()
st.title("chartcast")
st.caption("Time series chart → extracted data → forecast with an empirical probability of being wrong")


@st.cache_data(show_spinner=False)
def run_extract(data: bytes):
    return extract(data)


def show_problem(e: Exception, image: bytes | None = None):
    """Input problems get their own message; anything else is reported as an internal error.
    The message comes first so it is always visible; the image preview (if any) goes below it."""
    if isinstance(e, ChartError):
        st.warning(str(e))
    else:
        st.error("Something went wrong while processing this chart.")
        with st.expander("Technical details"):
            st.exception(e)
    if image is not None:
        img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        if img is not None and min(img.shape[:2]) >= 100:  # tiny images are not worth a preview
            st.image(image, caption="Your image", width=min(400, img.shape[1]))
    st.stop()


_POPUP_CSS = """<style>
.popable img{width:100%;display:block;cursor:zoom-in;border-radius:6px}
.popup{display:none;position:fixed;inset:0;z-index:999999;background:rgba(0,0,0,.78);
       align-items:center;justify-content:center}
.popup:target{display:flex}
.popup img{max-width:94vw;max-height:90vh;background:#fff;border-radius:6px;position:relative;z-index:1}
.popup .backdrop{position:absolute;inset:0;cursor:zoom-out}
.popup .x{position:absolute;top:14px;right:22px;z-index:2;color:#fff;font-size:34px;text-decoration:none;line-height:1}
</style>"""


def show_popable(rgb: np.ndarray, key: str, caption: str | None = None, box: tuple[int, int] | None = None):
    """A picture that opens as a large pop-up when clicked (pure HTML/CSS: no page reload).
    `box` = (width, height): show it fitted inside a box of that shape, so pictures side by side match."""
    uri = "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", rgb[..., ::-1])[1]).decode()
    fit = f' style="aspect-ratio:{box[0]}/{box[1]};object-fit:contain;background:#fff"' if box else ""
    st.markdown(_POPUP_CSS + f'<a class="popable" href="#pop-{key}"><img src="{uri}" alt="{key}"{fit}></a>'
                f'<div class="popup" id="pop-{key}"><a class="backdrop" href="#_"></a>'
                f'<a class="x" href="#_">×</a><img src="{uri}" alt="{key} enlarged"></div>',
                unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def to_rgb(fig, sharpness: float = 2.0) -> np.ndarray:
    """Render a matplotlib Figure to an RGB array, at `sharpness`× so it stays crisp when enlarged."""
    fig.set_dpi(fig.dpi * sharpness)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3]


# 1. Choose the image -------------------------------------------------------------------------
GROUPS = {"real": "Real screenshots", "synthetic": "Synthetic charts"}
IMAGE_TYPES = (".png", ".jpg", ".jpeg")


@st.cache_data
def thumbnail(path: str, width: int = 480) -> str:
    """Small PNG of a sample as a data URI, so the gallery stays light."""
    img = cv2.imread(path)
    img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", img)[1]).decode()


@st.dialog("Test samples", width="large")
def sample_gallery():
    """All test samples as clickable pictures: a click reloads the page with ?sample=<folder>/<file>."""
    style = ("<style>.gallery{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:18px}"
             ".gallery a{display:flex;align-items:center;border:1px solid rgba(128,128,128,.3);border-radius:8px;"
             "padding:6px;background:#fff}.gallery a:hover{border-color:#378ADD;box-shadow:0 0 0 2px #378ADD55}"
             ".gallery img{width:100%;display:block}</style>")
    st.markdown(style, unsafe_allow_html=True)
    for folder, title in GROUPS.items():
        files = sorted(p for p in (SAMPLES / folder).glob("*") if p.suffix.lower() in IMAGE_TYPES)
        if not files:
            continue
        st.subheader(title)
        cells = "".join(f'<a href="?sample={quote(f"{folder}/{p.name}")}" target="_self">'
                        f'<img src="{thumbnail(str(p))}" alt="{p.stem}"></a>' for p in files)
        st.markdown(f'<div class="gallery">{cells}</div>', unsafe_allow_html=True)


def chosen_sample() -> Path | None:
    """The sample picked in the gallery (from the URL), only if it really is a file in samples/."""
    rel = st.query_params.get("sample")
    if not rel:
        return None
    path = (SAMPLES / rel).resolve()
    ok = path.is_file() and path.suffix.lower() in IMAGE_TYPES and SAMPLES.resolve() in path.parents
    return path if ok else None


source = st.radio("Image", ["Test sample", "Upload your own"], horizontal=True, label_visibility="collapsed")
if source == "Test sample":
    if st.button("Choose a test sample", icon=":material/photo_library:"):
        sample_gallery()
    path = chosen_sample()
    if path is None:
        st.info("Click **Choose a test sample** to browse the samples and pick one.")
        st.stop()
    data, name = path.read_bytes(), f"{path.parent.name} / {path.name}"
    truth_file = path.with_suffix(".json")  # synthetic samples come with the data they were drawn from
else:
    uploaded = st.file_uploader("Chart image", type=["png", "jpg", "jpeg"])
    if uploaded is None:
        st.info("Upload a line chart image (PNG or JPG) to start.")
        st.stop()
    data, name = uploaded.getvalue(), uploaded.name
    truth_file = None
st.caption(f"Selected: **{name}**")

loading = ("Reading the chart…" if ocr_ready()
           else "Reading the chart… (the first chart after starting the app also loads the OCR model, up to ~30 s)")
try:
    with st.spinner(loading):
        ex = run_extract(data)
except Exception as e:
    show_problem(e, data)
s = ex.series
TFMT = time_format(s.index, ex.freq)
when = lambda t: t.strftime(TFMT)

st.header("1. Time series graph reconstruction")
original = cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
box = (original.shape[1], original.shape[0])  # both pictures in boxes of the original's shape
left, right = st.columns(2)
with left:
    show_popable(original, "original", "Original", box)
with right:
    # the input image is upscaled internally: draw at the prepared size, shown in the original's box
    show_popable(to_rgb(reconstruct(ex, size=ex.prep.gray.shape[::-1])), "reconstruction",
                 "Reconstructed: only the axes, the labels and the traced line", box)
if truth_file is not None and truth_file.exists():
    picture, err = accuracy_check(s, truth_file)
    show_popable(picture, "accuracy", f"Accuracy check (synthetic sample, true data known): the extracted series "
                                      f"(blue) over the true one (grey). Mean error {err:.1%} of the value range.")
with st.expander("Extraction details / Download CSV"):
    show_popable(overlay(ex), "overlay",
                 "Plot box (orange), value labels (green), time labels (blue), axis lines (red), traced line (magenta)")
    st.markdown(f"<details><summary><b>Details</b></summary><pre>"
                f"{html.escape(json.dumps(ex.confidence, indent=2, ensure_ascii=False))}</pre></details>",
                unsafe_allow_html=True)
    st.download_button("Download extracted CSV", s.to_csv().encode(), "extracted.csv", "text/csv")

st.header("2. Graph overview")
step = FREQS[ex.freq]
change = (s.iloc[-1] - s.iloc[0]) / abs(s.iloc[0]) if s.iloc[0] else float("nan")
stat_row([("Sampling", FREQ_NAMES.get(ex.freq, ex.freq), FREQ_SHORT.get(ex.freq)),
          ("Samples", str(len(s)), None),
          ("From", when(s.index[0]), None),
          ("To", when(s.index[-1]), None),
          ("Value range", f"{s.min():.4g} – {s.max():.4g}", None)])
stat_row([("Time span", duration((s.index[-1] - s.index[0]).total_seconds() + step), None)]
         + [("Initial value", f"{s.iloc[0]:.4g}", None),
            ("Ending value", f"{s.iloc[-1]:.4g}", None),
            ("Overall change", f"{change:+.1%}", None),
            ("Scale", "log" if ex.confidence["y_calibration"]["log"] else "linear", None)])
st.caption(f"The chart is sampled **{FREQ_NAMES.get(ex.freq, ex.freq)}**: {PER_STEP.get(ex.freq, 'one value per step')}, "
           f"{len(s)} samples from {when(s.index[0])} to {when(s.index[-1])}.")

# 3. Forecast settings ------------------------------------------------------------------------
st.header("3. Forecast settings")
units = possible_units(ex.freq)
unit0 = DEFAULT_UNIT.get(ex.freq, units[0])
st.info(f"This chart is sampled **{FREQ_NAMES.get(ex.freq, ex.freq)}**, so choose the horizon, the training "
        f"window and the step in {', '.join(units[:-1]) + ' or ' + units[-1] if len(units) > 1 else units[0]}.")
min_amount = lambda unit, samples: max(1, int(np.ceil(samples * FREQS[ex.freq] / UNITS[unit] - 1e-9)))

model = st.selectbox("Forecasting model", list(MODEL_NAMES), format_func=MODEL_NAMES.get)
a, b = st.columns(2)
h_unit = b.selectbox("Horizon unit", units, index=units.index(unit0))
h_min = min_amount(h_unit, 1)
h_amount = a.number_input("Forecast the next", min_value=h_min, value=h_min, step=1)
a, b = st.columns(2)
w_unit = b.selectbox("Training window unit", units, index=units.index(unit0))
w_min = min_amount(w_unit, 2)
half = max(w_min, int(len(s) / 2 * FREQS[ex.freq] / UNITS[w_unit]))  # default window ≈ half the history
w_amount = a.number_input("Train on the last", min_value=w_min, value=half, step=1)
horizon, window = to_steps(h_amount, h_unit, ex.freq), to_steps(w_amount, w_unit, ex.freq)

# The window slides through the history by `step`: at least 1 sample, at most what still leaves 2 folds.
a, b = st.columns(2)
s_unit = b.selectbox("Step unit", units, index=0)  # finest unit, so the default can give ~20 folds
s_min = min_amount(s_unit, 1)
room = len(s) - window - horizon                     # samples the window can slide over
s_default = max(s_min, int(max(room, 1) / 19 * FREQS[ex.freq] / UNITS[s_unit]))  # ≈ 20 folds
s_amount = a.number_input("Move the window by (step)", min_value=s_min, value=s_default, step=1)
step = to_steps(s_amount, s_unit, ex.freq)
tolerance = st.slider("Tolerance: a forecast is 'incorrect' when it misses by more than", 1, 50, 10,
                      format="%d%%") / 100
run = st.button("Run forecast", type="primary")

plural = lambda n, unit: f"{n} {unit[:-1] if n == 1 else unit}"
n_folds = len(fold_origins(len(s), horizon, window, step)) if room >= 0 and step >= 1 else 0
limits = (f"step can be 1 to {room} samples" if room >= 1 else "no room to slide: shorten the window or horizon")
samples = lambda k: f"**{k}** sample{'' if k == 1 else 's'}"
st.caption(f"{plural(h_amount, h_unit)} = {samples(horizon)}, window {plural(w_amount, w_unit)} = {samples(window)}, "
           f"step {plural(s_amount, s_unit)} = {samples(step)} → **{n_folds}** validation fold{'' if n_folds == 1 else 's'} "
           f"({limits}; at most {MAX_FOLDS} folds), data is {FREQ_NAMES.get(ex.freq, ex.freq)} with {len(s)} samples")
key = (name, len(data), model, horizon, window, step, tolerance)
if run:
    try:
        check_window(len(s), horizon, window, step)
        with st.spinner("Rolling validation…"):
            compare = model in ("auto_error", "best_pinc")  # validate all models and pick one
            criterion = "p_incorrect" if model == "best_pinc" else "mean_error"
            st.session_state["result"] = (key, *select_and_forecast(s, horizon, ex.period, window,
                                                                    MODELS if compare else (model,), step,
                                                                    criterion, tolerance))
    except Exception as e:
        st.session_state.pop("result", None)
        show_problem(e)
if st.session_state.get("result", (None,))[0] != key:
    st.stop()

# 4. Results ----------------------------------------------------------------------------------
_, fc, best, results = st.session_state["result"]
report = error_report(best, tolerance)
st.header("4. Forecast")
c = st.columns(4)
c[0].metric("Model", MODEL_NAMES[best.model].split(" (")[0],
            help={"auto_error": "Chosen automatically: lowest mean error in validation",
                  "best_pinc": "Chosen by lowest P(incorrect) in validation"}.get(model))
c[1].metric("Mean error baseline", f"{report.mean_error:.1%}")
c[2].metric(f"Forecast at {when(fc.index[-1])}", f"{fc.iloc[-1]:.4g}")
c[3].metric("P(incorrect) at horizon end", f"{report.p_incorrect[-1]:.0%}",
            help=f"95% interval {report.ci_low[-1]:.0%} – {report.ci_high[-1]:.0%}")

fig = Figure(figsize=(11, 4), dpi=100)
ax = fig.add_subplot()
ax.plot(s.index, s.values, color="0.55", lw=1, label="extracted")
ax.plot(s.index[-window:], s.values[-window:], color="0.15", lw=1.4, label="training window")
ax.plot([s.index[-1], *fc.index], [s.iloc[-1], *fc.values], color="#378ADD", lw=2, label="forecast")
ax.legend(loc="upper left", frameon=False)
ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=9))
ax.xaxis.set_major_formatter(mdates.DateFormatter(TFMT))  # same format as From/To (no placeholder dates)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
show_popable(to_rgb(fig), "forecast")

st.caption(f"Rolling validation: {len(best.origins)} folds, {plural(s_amount, s_unit)} apart; each trains on the "
           f"{plural(w_amount, w_unit)} before a point in the past, forecasts the next {plural(h_amount, h_unit)}, "
           f"and is compared with what the chart shows. P(incorrect) = share of folds missing by more than "
           f"{tolerance:.0%} (95% Wilson interval).")
st.dataframe(pd.DataFrame({
    "time": [when(t) for t in fc.index], "forecast": fc.values.round(4),
    "P(incorrect)": report.p_incorrect.round(3), "CI low": report.ci_low.round(3), "CI high": report.ci_high.round(3),
}), hide_index=True, width="stretch")
if len(results) > 1:
    st.subheader("Models compared in validation")
    st.caption(f"Each model was validated the same way. **Mean error baseline:** its average relative error over all "
               f"folds and steps. **P(incorrect):** the share of its forecasts (all folds and steps) that missed by "
               f"more than {tolerance:.0%}. Lower is better for both; the chosen model is marked.")
    st.dataframe(pd.DataFrame({
        "model": [MODEL_NAMES[m].split(" (")[0] + ("  ← chosen" if m == best.model else "") for m in results],
        "mean error baseline": [f"{b.mean_error:.2%}" for b in results.values()],
        "P(incorrect)": [f"{b.p_incorrect(tolerance):.0%}" for b in results.values()],
    }), hide_index=True, width="stretch")
