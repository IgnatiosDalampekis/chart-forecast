# chartcast

Upload an image of a time series chart → extract the plotted data with computer vision and OCR → forecast it →
estimate, from rolling forecast validation, the mean error baseline and the probability that the forecast is wrong.

Fully automatic: the only input is the image. One pipeline for every chart, synthetic or real.

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The EasyOCR models (~100 MB) download on first use into `.models/` (git-ignored). The app loads them in the
background as soon as it opens, so the first chart does not wait for them.

```bash
pytest                 # all tests (~1 min after the OCR models are downloaded)
pytest -m "not ocr"    # fast unit tests only
python -m chartcast.synth --n 5 --out samples/synthetic   # regenerate synthetic sample charts
```

## The app

1. **Choose an image:** click *Choose a test sample* and pick a picture from the gallery, or upload your own chart.
2. **Time series graph reconstruction:** the original next to the reconstruction (same size), which keeps only the
   axes, the labels that were read and the traced line. For synthetic samples, an accuracy check draws the true data
   with the extracted series on top and gives the mean error. *Extraction details / Download CSV* (collapsed) holds
   the detection overlay, the per-stage checks (*Details*, collapsed) and the CSV download.
   Click any picture to open it larger.
3. **Graph overview:** sampling, number of samples, from / to, value range, time span, initial and ending value,
   overall change and scale (linear or log), plus a sentence describing the sampling.
4. **Forecast settings:**
   - **Model:** *Auto (lowest mean error in validation)*, *Lowest P(incorrect) in validation*, or one model directly
     (naive, seasonal naive, Holt-Winters, ARIMA).
   - **Horizon**, **training window** and **step** (how far the window moves between validation folds), each as an
     amount plus a unit. Only units possible for the chart's sampling are offered (e.g. days, weeks, months, years
     for daily data), and durations are converted to samples (1 day of hourly data = 24 samples). A live line shows
     the resulting number of validation folds and the allowed step range.
   - **Tolerance τ:** a forecast counts as incorrect when it misses by more than τ.
5. **Forecast:** the model used, the mean error baseline, the forecast at the end of the horizon and P(incorrect)
   there; the forecast drawn as a continuation of the line; a table per future step (forecast, P(incorrect) and its
   95% interval); and, when models are compared, each model's mean error baseline and P(incorrect).

## When an image can't be used

Problems caused by the input stop the pipeline with a message for the user (`chartcast/errors.py`): unreadable file,
image too small, no readable text, **no value labels** ("Please upload an image with value data…"), **no time
labels** ("Please upload an image with time data…"), no data line, too few data points to forecast, and horizon /
window / step settings that do not fit the extracted series. Anything else is reported as an internal error with
technical details.

## Repository layout

```
chartcast/
├── config.py       # pipeline settings (one set for every chart)
├── errors.py       # input errors with user-facing messages
├── preprocess.py   # load, upscale, dark-theme handling, ink masks
├── ocr.py          # EasyOCR: all text in the image, word merging, axis-strip re-read, background warm-up
├── parse.py        # text -> number / timestamp (broad format coverage)
├── axes.py         # value axis + time axis from the labels, aligned axis lines, plot box
├── calibrate.py    # pixel -> value (linear default, log on evidence) / pixel -> time
├── line.py         # data-line isolation inside the plot box
├── trace.py        # skeleton tracing, frequency, resampling -> regular time series
├── forecast.py     # models, sliding-window rolling validation, model selection
├── uncertainty.py  # P(incorrect) per step with Wilson intervals
├── pipeline.py     # end-to-end extraction, overlay and reconstruction drawings
└── synth.py        # synthetic charts with known ground truth
app.py              # Streamlit UI
samples/real/       # real chart screenshots
samples/synthetic/  # synthetic charts, each with a JSON of the true data it was drawn from
tests/              # unit tests + accuracy tests against synthetic ground truth
```

## Extraction pipeline

| # | Stage | Methods | Output |
|---|---|---|---|
| 1 | Preprocess | Upscale small images, dark-theme inversion, Otsu binarisation, HSV ink masks (neutral vs coloured) | Normalised image + masks |
| 2 | OCR pass 1 | EasyOCR on the whole image (2× enlarged); words on one line with similar height merged (`Sep` + `2023`); OCR misread cleanup | All text with positions |
| 3 | Value axis | Numbers aligned on one vertical line (left or right), increasing bottom → top, on one common scale; confident labels anchor the scale, doubtful ones are cross-checked by interpolation (kept, corrected if they sit on the tick grid between anchors, otherwise dropped); ≥ 3 labels preferred, 2 accepted; an axis line running along the labels breaks ties | Value labels, side, axis line if drawn |
| 4 | Time axis | Dates/times on one horizontal line, increasing left → right (year-less labels unwrapped), same one-scale cross-check | Time labels, axis line if drawn |
| 5 | OCR pass 2 + box | Each label strip re-read enlarged (numbers only for the value strip), used if it reads more labels; box from the axis lines when present, else from the labels; the traced line sets the final horizontal extent | Plot box |
| 6 | Calibration | Label centres snapped to tick marks or gridlines; robust least squares pixel → value and pixel → timestamp; **linear by default**, log when the chart says "log" or the labels fit log10 clearly better | `pixel → value`, `pixel → time` |
| 7 | Line isolation | Inside the box: mask text and gridlines, cluster ink in Lab colour space (fixed seed) | Line mask |
| 8 | Tracing | Skeletonisation + spur pruning, per-column selection with continuity, sub-pixel refinement, peak/trough correction | Pixel trace |
| 9 | Time series | Sampling frequency from the FFT of the line's corners (else the finest step the plot width can resolve; sub-hourly only with clock-time labels), resampling to one value per step, seasonal period from autocorrelation | `pd.Series` |

### Why text first

Axis lines, gridlines and borders are decoration: many charts omit them and screenshots add unrelated ones. Tick
labels are what make a chart readable, so every chart has them: a column of numbers increasing upwards and a row of
dates increasing to the right, on whichever side they are drawn. Lines only confirm a label group and refine the box.

### Why two OCR passes

Pass 1 reads the whole image to *find* the axis labels among all the text. Once their location is known, pass 2
re-reads just those strips enlarged (digits only for the values), which recovers small, glued or misread labels.
On the test charts pass 2 improved the value labels on about half of them.

### Recognised label formats

- **Numbers:** integers, decimals, negatives (`-3`, `−3`, `(3)`), thousands separators (`1,234` `1.234` `1 234`
  `1'234`), decimal comma, `%`, currencies (`$ € £ ¥ ₹`, codes), scale suffixes (`k M B bn T`), scientific (`1e5`,
  `10³`, `2×10^3`), attached units (`°C`, `MW`, `kWh`).
- **Time:** ISO dates and datetimes, numeric dates (`05/01/2024`, `5.1.24`), month names (full/short, with day
  and/or year), years (`2024`, `'24`), quarters, halves, fiscal years, weeks, weekdays, clock times (`14:00`, `2pm`,
  `14h`, also OCR-misread `12.00 p.m.`) and combinations.

### Sampling frequency

Line charts draw straight segments between samples, so the line turns at every sample and the turns sit on a regular
pixel grid. The FFT of the line's second difference finds that spacing; multiplied by the time per pixel (from the
time-axis calibration) and snapped to a standard step (1/5/15/30 min, hourly, daily, weekly, monthly, quarterly,
yearly), it gives the frequency. When the corners are too dense to see, the finest step the plot width can resolve
is used.

## Forecasting and validation

| Step | What happens |
|---|---|
| Models | Naive (last value), seasonal naive (last cycle), Holt-Winters (damped trend, seasonal when the window holds 2+ cycles), ARIMA (SARIMAX(1,1,1), seasonal term when the window is long enough) |
| Rolling validation | Each fold trains on exactly the chosen window of past samples, forecasts the horizon, and is compared with what the chart shows; folds are `step` samples apart, laid back from the most recent data (default ≈ 20 folds, at least 2, at most 200) |
| Selection | *Auto*: lowest mean error baseline; or lowest P(incorrect); or a fixed model |
| Final forecast | The chosen model refit on the last window, forecasting the horizon |

### The numbers

With `e(k, h) = |actual − forecast| / |actual|` the relative error of fold `k` at step `h` ahead:

```
mean error baseline = average of e(k, h) over all folds k and steps h
P(incorrect | h)    = (number of folds k with e(k, h) > τ) / (number of folds)
```

P(incorrect) is reported per future step with a 95% **Wilson interval**, because it is estimated from a limited
number of folds (e.g. 0 of 20 failures → 0%, plausibly up to ~16%).

## Measured accuracy (synthetic charts, true data known)

| Check | Result |
|---|---|
| Axes found (value + time) | 20/20 charts |
| Plot box (vertical) vs ground truth | within 2 px |
| Y calibration (value at true pixel row) | < 0.2 % of axis range |
| X calibration (pixel of true timestamp) | < 0.7 px |
| Frequency detection | 18/20 (smooth monthly charts can be read as weekly) |
| Series error | median 1.5 % of the value range |

The real screenshots in `samples/real/` (stock charts from different sources, daily, weekly and intraday) are checked
visually with the reconstruction and the detection overlay.

## Determinism

The pipeline is fully deterministic: running the same experiment on the same image always gives exactly the same
results (the same extracted series, model choice, forecast, mean error baseline and P(incorrect)). Every random
component has a fixed seed (colour clustering, robust fits), and OCR and the forecasting models are deterministic.
Verified by running the same charts in separate processes: identical results on every run.

## Known issues

- Smooth monthly series can give too weak a corner signal and be read as weekly.
- Charts whose labels have no year (`Jan 05`) or only clock times get a placeholder date (year 2000), shown without it.
- The relative error `|a − f| / |a|` explodes for series close to zero.
- Validation folds overlap (they share training data), so the Wilson interval is somewhat optimistic.
- Daily stock charts have no weekends; the series is resampled to every calendar day.
- Very steep noise spikes are limited by pixel resolution; thick black lines can start a few samples late.

## Scope and limitations

Line charts only. Not supported: bar and candlestick charts, overlapping same-colour lines, 3D or rotated charts,
charts without readable tick labels.
